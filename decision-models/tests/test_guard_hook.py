"""`agents/guard_hook.py`, the campaign's one guard (typed-decision-models U5.7; plan6
`docs/plans/typed-decision-models6.md` §3.3 "Guard decisions" and the §5 U5.7 row).

The guard runs as the process Claude Code starts, from an installed campaign (a real `driver
install` into `tmp_path`): `stdin` JSON in, one decision line and exit 0, or exit 1 with nothing
on stdout. The bounds axes of P3-m3 are planted cases, each with its expected decision written
here: relative or absolute × `..` × a symlinked file or directory × a sibling prefix (`ans-x/`) ×
the bound directory itself × an empty or non-string path. The prompt the guard allows is compared
with `Serve.spawn_prompt` from the campaign's own `next`, so the guard and the runner agree.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from test_audit import GUARD as FIXTURE_GUARD
from test_install import GUARD_COMMAND, Installed, installed

from decisionmodels import audit
from decisionmodels.campaign import Serve

pytestmark = pytest.mark.usefixtures("fake_claude")
GUARD = Path(__file__).resolve().parents[1] / "agents" / "guard_hook.py"


@dataclass
class Tree:
    made: Installed
    exchange: Path
    serve: Serve

    @property
    def guard(self) -> Path:
        return self.exchange / ".claude" / "guard_hook.py"

    @property
    def log(self) -> Path:
        return self.exchange.parent / "store" / "hook-log.jsonl"

    def run(self, payload: Any = None, *, raw: str | None = None) -> tuple[int, str, str]:
        text = raw if raw is not None else json.dumps(payload)
        done = subprocess.run([sys.executable, str(self.guard)], input=text, cwd=self.exchange,
                              capture_output=True, text=True)
        return done.returncode, done.stdout, done.stderr

    def decision(self, payload: dict[str, Any]) -> str:
        code, out, err = self.run(payload)
        assert code == 0 and err == "", (code, err)
        specific = json.loads(out)["hookSpecificOutput"]
        assert specific["hookEventName"] == "PreToolUse" and specific["permissionDecisionReason"]
        return specific["permissionDecision"]

    def call(self, agent: Any, tool: str, tool_input: Any, **extra: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {"hook_event_name": "PreToolUse", "agent_type": agent,
                                   "tool_name": tool, "tool_input": tool_input,
                                   "tool_use_id": "toolu_t1", "cwd": str(self.exchange)}
        payload.update(extra)
        return {k: v for k, v in payload.items() if v is not _MISSING}


_MISSING = object()


@pytest.fixture
def tree(tmp_path: Path) -> Tree:
    made = installed(tmp_path)
    camp = made.camp
    serve = camp.next()
    assert isinstance(serve, Serve)
    camp.issue(serve)
    return Tree(made, camp.exchange_dir, serve)


def agent_input(tree: Tree, **edits: Any) -> dict[str, Any]:
    base = {"subagent_type": "dm-labeler", "prompt": tree.serve.spawn_prompt,
            "description": "dm batch"}
    base.update(edits)
    return {k: v for k, v in base.items() if v is not _MISSING}


# --- the Agent call of the driver ----------------------------------------------------------------


def test_the_exact_pending_spawn_is_allowed_and_matches_the_campaigns_own_prompt(
        tree: Tree) -> None:
    assert tree.serve.spawn_prompt.startswith(f"Read {tree.exchange}/req/")
    assert tree.decision(tree.call("dm-driver", "Agent", agent_input(tree))) == "allow"


def test_the_description_may_be_absent(tree: Tree) -> None:
    assert tree.decision(tree.call("dm-driver", "Agent",
                                   agent_input(tree, description=_MISSING))) == "allow"


@pytest.mark.parametrize("extra", [
    {"effort": "low"}, {"model": "opus"}, {"isolation": "worktree"},
    {"run_in_background": False}, {"run_in_background": True}, {"mode": "bypassPermissions"},
    {"name": "x"}, {"resume": "a1"}])
def test_an_agent_call_with_a_key_outside_the_three_is_denied(tree: Tree,
                                                              extra: dict[str, Any]) -> None:
    assert tree.decision(tree.call("dm-driver", "Agent", agent_input(tree, **extra))) == "deny"


@pytest.mark.parametrize("edits", [
    {"subagent_type": "dm-generator"},        # the pending request is a label request
    {"subagent_type": "general-purpose"},
    {"subagent_type": "fork"},
    {"subagent_type": "dm-driver"},
    {"subagent_type": None},
    {"subagent_type": _MISSING},
    {"prompt": _MISSING},
    {"prompt": 7},
    {"description": 7},
])
def test_an_agent_call_with_another_subagent_type_or_a_broken_field_is_denied(
        tree: Tree, edits: dict[str, Any]) -> None:
    assert tree.decision(tree.call("dm-driver", "Agent", agent_input(tree, **edits))) == "deny"


def test_a_prompt_that_is_not_the_pending_spawn_prompt_is_denied(tree: Tree) -> None:
    prompt = tree.serve.spawn_prompt
    for wrong in (prompt + " ", prompt + "\n", prompt[:-1], prompt.replace("Read", "Write", 1),
                  prompt.replace(".jsonl", "x.jsonl", 1), "", "done"):
        assert tree.decision(tree.call("dm-driver", "Agent",
                                       agent_input(tree, prompt=wrong))) == "deny", wrong


def test_a_prompt_for_a_request_that_is_not_pending_is_denied(tree: Tree) -> None:
    other = tree.serve.spawn_prompt.replace(tree.serve.opaque, "qqqqqqqqqqqqqqqq")
    assert tree.decision(tree.call("dm-driver", "Agent", agent_input(tree, prompt=other))) == "deny"


def test_no_pending_request_and_two_pending_requests_are_denied(tree: Tree) -> None:
    request = tree.exchange / "req" / f"{tree.serve.opaque}.jsonl"
    second = tree.exchange / "req" / "second.jsonl"
    shutil.copy(request, second)
    assert tree.decision(tree.call("dm-driver", "Agent", agent_input(tree))) == "deny"
    second.unlink()
    request.unlink()
    assert tree.decision(tree.call("dm-driver", "Agent", agent_input(tree))) == "deny"


def test_a_generation_request_allows_only_the_generator(tree: Tree) -> None:
    request = tree.exchange / "req" / f"{tree.serve.opaque}.jsonl"
    request.unlink()
    generation = tree.exchange / "req" / "g1.jsonl"
    generation.write_text(json.dumps({"role": "generate"}) + "\n", "utf-8")
    template = (tree.exchange / ".claude" / "spawn-generator.txt").read_text("utf-8")
    prompt = template.replace("{request_path}", str(generation))
    assert tree.decision(tree.call("dm-driver", "Agent", agent_input(
        tree, subagent_type="dm-generator", prompt=prompt))) == "allow"
    assert tree.decision(tree.call("dm-driver", "Agent", agent_input(
        tree, subagent_type="dm-labeler", prompt=prompt))) == "deny"


def test_an_unknown_role_in_the_request_header_denies(tree: Tree) -> None:
    request = tree.exchange / "req" / f"{tree.serve.opaque}.jsonl"
    request.write_text(json.dumps({"role": "other"}) + "\n", "utf-8")
    assert tree.decision(tree.call("dm-driver", "Agent", agent_input(tree))) == "deny"


@pytest.mark.parametrize("tool", ["Read", "Write", "Bash", "Edit", "Glob", "WebFetch",
                                  "SubagentHandback"])
def test_the_driver_may_call_only_agent(tree: Tree, tool: str) -> None:
    inside = {"file_path": str(tree.exchange / "req" / "x.jsonl"), "command": "ls"}
    assert tree.decision(tree.call("dm-driver", tool, inside)) == "deny"
    # Even with the exact input of the one allowed Agent call.
    assert tree.decision(tree.call("dm-driver", tool, agent_input(tree))) == "deny"


def test_a_subagent_may_not_call_agent(tree: Tree) -> None:
    for agent in ("dm-labeler", "dm-generator"):
        assert tree.decision(tree.call(agent, "Agent", agent_input(tree))) == "deny"


# --- the subagents' bounds -----------------------------------------------------------------------


def paths(tree: Tree, tmp_path: Path) -> None:
    e = tree.exchange
    (e.parent / "store").mkdir(exist_ok=True)
    (e / "req" / "inner.jsonl").write_text("x", "utf-8")
    (e / "ans-x").mkdir()
    (e / "req-x").mkdir()
    (tmp_path / "outside.txt").write_text("secret", "utf-8")
    (e / "req" / "to-outside").symlink_to(tmp_path / "outside.txt")
    (e / "req" / "to-inner").symlink_to(e / "req" / "inner.jsonl")
    (e / "ans" / "to-store").symlink_to(e.parent / "store" / "ledger.jsonl")
    (e / "req" / "dir-to-store").symlink_to(e.parent / "store")
    (e / "ans" / "dir-to-req").symlink_to(e / "req")
    (e / "via-req").symlink_to(e / "req")


@pytest.mark.parametrize("agent", ["dm-labeler", "dm-generator"])
def test_read_bounds(tree: Tree, tmp_path: Path, agent: str) -> None:
    paths(tree, tmp_path)
    e = str(tree.exchange)
    cases: list[tuple[str, str | None, str, str]] = [
        # (file_path, cwd, expected, why)
        (f"{e}/req/x.jsonl", e, "allow", "absolute inside"),
        ("req/x.jsonl", e, "allow", "relative inside, joined to cwd"),
        ("x.jsonl", f"{e}/req", "allow", "relative to a cwd that is the bound"),
        (f"{e}/req/../req/x.jsonl", e, "allow", ".. that stays inside"),
        (f"{e}/ans/../req/x.jsonl", e, "allow", ".. that lands inside"),
        (f"{e}/req/to-inner", e, "allow", "symlinked file that resolves inside"),
        (f"{e}/via-req/x.jsonl", e, "allow", "symlinked directory that resolves inside"),
        (f"{e}/req/../ans/x.jsonl", e, "deny", ".. into ans"),
        (f"{e}/req/../../store/ledger.jsonl", e, "deny", ".. out of the exchange"),
        ("../store/ledger.jsonl", e, "deny", "relative .. out"),
        ("../ans/x.jsonl", f"{e}/req", "deny", "relative .. from the bound into ans"),
        (f"{e}/ans/x.jsonl", e, "deny", "Read of ans"),
        (f"{e}/.claude/guard_hook.py", e, "deny", "the guard itself"),
        (f"{e}/req/to-outside", e, "deny", "symlinked file that resolves outside"),
        (f"{e}/req/dir-to-store/ledger.jsonl", e, "deny", "symlinked directory outside"),
        (f"{e}/req-x/y.jsonl", e, "deny", "sibling prefix"),
        (f"{e}/req", e, "deny", "the bound itself"),
        (f"{e}/req/", e, "deny", "the bound with a slash"),
        (f"{e}/req/.", e, "deny", "the bound as dot"),
        ("req", e, "deny", "the bound, relative"),
        (".", f"{e}/req", "deny", "dot in the bound"),
        ("/etc/passwd", e, "deny", "absolute elsewhere"),
        ("x.jsonl", None, "deny", "relative with no cwd"),
        ("x.jsonl", "req", "deny", "relative with a relative cwd"),
        ("", e, "deny", "empty"),
        ("", f"{e}/req/sub", "deny", "empty, from a cwd inside the bound"),
        (f"{e}/req/\x00x", e, "deny", "an embedded NUL"),
    ]
    for file_path, cwd, expected, why in cases:
        payload = tree.call(agent, "Read", {"file_path": file_path},
                            cwd=cwd if cwd is not None else _MISSING)
        assert tree.decision(payload) == expected, why


@pytest.mark.parametrize("agent", ["dm-labeler", "dm-generator"])
def test_write_bounds(tree: Tree, tmp_path: Path, agent: str) -> None:
    paths(tree, tmp_path)
    e = str(tree.exchange)
    cases: list[tuple[str, str, str]] = [
        (f"{e}/ans/x.jsonl", "allow", "absolute inside"),
        (f"{e}/ans/x.1.jsonl", "allow", "a part"),
        ("ans/x.jsonl", "allow", "relative inside"),
        (f"{e}/ans/../ans/x.jsonl", "allow", ".. that stays inside"),
        (f"{e}/ans/dir-to-req/x.jsonl", "deny", "symlinked directory into req"),
        (f"{e}/ans/to-store", "deny", "symlinked file into the store"),
        (f"{e}/ans/../req/x.jsonl", "deny", ".. into req"),
        (f"{e}/ans/../../store/x", "deny", ".. out"),
        (f"{e}/req/x.jsonl", "deny", "Write of req"),
        (f"{e}/ans-x/y.jsonl", "deny", "sibling prefix"),
        (f"{e}/ans", "deny", "the bound itself"),
        (f"{e}/ans/", "deny", "the bound with a slash"),
        ("ans", "deny", "the bound, relative"),
        (f"{e}/.claude/guard_hook.py", "deny", "the guard itself"),
        (f"{e}/.claude/settings.json", "deny", "the guard's wiring"),
        ("/tmp/x", "deny", "absolute elsewhere"),
        ("", "deny", "empty"),
    ]
    for file_path, expected, why in cases:
        payload = tree.call(agent, "Write", {"file_path": file_path, "content": "x"})
        assert tree.decision(payload) == expected, why


@pytest.mark.parametrize("tool", ["Read", "Write"])
@pytest.mark.parametrize("value", [None, 5, True, ["a"], {"a": 1}, 1.5])
def test_a_non_string_or_missing_path_is_denied(tree: Tree, tool: str, value: Any) -> None:
    assert tree.decision(tree.call("dm-labeler", tool, {"file_path": value})) == "deny"
    assert tree.decision(tree.call("dm-labeler", tool, {})) == "deny"


def test_a_relative_path_uses_the_inputs_cwd_not_the_guards(tree: Tree, tmp_path: Path) -> None:
    other = tmp_path / "other"
    (other / "req").mkdir(parents=True)
    payload = tree.call("dm-labeler", "Read", {"file_path": "req/x.jsonl"}, cwd=str(other))
    assert tree.decision(payload) == "deny"          # the guard runs in exchange/, the call did not


@pytest.mark.parametrize("tool", ["Bash", "Edit", "MultiEdit", "Glob", "Grep", "WebFetch",
                                  "WebSearch", "NotebookEdit", "TodoWrite", "mcp__x__y"])
def test_a_subagent_may_call_only_read_write_and_the_hand_back(tree: Tree, tool: str) -> None:
    inside = {"file_path": str(tree.exchange / "req" / "x.jsonl"), "command": "ls"}
    for agent in ("dm-labeler", "dm-generator"):
        assert tree.decision(tree.call(agent, tool, inside)) == "deny"


def test_the_hand_back_is_allowed_for_the_two_subagents_only(tree: Tree) -> None:
    assert tree.decision(tree.call("dm-labeler", "SubagentHandback", {})) == "allow"
    assert tree.decision(tree.call("dm-generator", "SubagentHandback", {"x": 1})) == "allow"
    assert tree.decision(tree.call("dm-driver", "SubagentHandback", {})) == "deny"


# --- fail closed ---------------------------------------------------------------------------------


@pytest.mark.parametrize("edit", [
    {"agent_type": _MISSING}, {"agent_type": "general-purpose"}, {"agent_type": ""},
    {"agent_type": None}, {"agent_type": ["dm-driver"]}, {"agent_type": "DM-DRIVER"},
    {"tool_input": _MISSING}, {"tool_input": None}, {"tool_input": "x"}, {"tool_input": []},
    {"tool_name": _MISSING}, {"tool_name": 3},
    {"hook_event_name": "PostToolUse"}, {"hook_event_name": _MISSING},
])
def test_an_input_the_guard_cannot_judge_exits_1_with_no_decision(tree: Tree,
                                                                  edit: dict[str, Any]) -> None:
    good = {"file_path": str(tree.exchange / "req" / "x")}
    payload = tree.call("dm-labeler", "Read", edit.get("tool_input", good),
                        **{k: v for k, v in edit.items() if k != "tool_input"})
    code, out, err = tree.run(payload)
    assert (code, out) == (1, "") and err.startswith("guard failure")


@pytest.mark.parametrize("agent", ["dm-driver", "dm-labeler"])
@pytest.mark.parametrize("tool", ["SubagentHandback", "Bash"])
@pytest.mark.parametrize("tool_input", [_MISSING, None, "x"])
def test_a_tool_use_without_an_object_input_exits_1_whatever_the_tool(
        tree: Tree, agent: str, tool: str, tool_input: Any) -> None:
    code, out, _ = tree.run(tree.call(agent, tool, tool_input))
    assert (code, out) == (1, "")


@pytest.mark.parametrize("raw", ["", "not json", "[]", "null", '"x"', "{"])
def test_stdin_that_is_not_a_json_object_exits_1_with_no_decision(tree: Tree, raw: str) -> None:
    code, out, _ = tree.run(raw=raw)
    assert (code, out) == (1, "")


def test_a_failed_log_append_exits_1_with_no_decision(tree: Tree) -> None:
    shutil.rmtree(tree.exchange.parent / "store")
    payload = tree.call("dm-labeler", "Read", {"file_path": str(tree.exchange / "req" / "x")})
    code, out, err = tree.run(payload)
    assert (code, out) == (1, "") and "FileNotFoundError" in err


def load_guard() -> Any:
    spec = importlib.util.spec_from_file_location("guard_under_test", GUARD)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_an_exception_from_the_decision_exits_1_with_no_decision(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    guard = load_guard()

    def boom(data: Any, exchange: Path) -> tuple[str, str]:
        raise RuntimeError("planted")
    monkeypatch.setattr(guard, "decide", boom)
    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO("{}"))
    assert guard.main() == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "RuntimeError: planted" in captured.err


def test_a_planted_exception_inside_a_branch_is_caught_too(
        tree: Tree, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """`os.path.realpath` failing is not a deny: it is a failure the wiring blocks."""
    guard = load_guard()
    monkeypatch.setattr(guard, "exchange_dir", lambda: tree.exchange)

    def boom(path: str) -> str:
        raise OSError("planted")
    monkeypatch.setattr(guard.os.path, "realpath", boom)
    payload = tree.call("dm-labeler", "Read", {"file_path": str(tree.exchange / "req" / "x")})
    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO(json.dumps(payload)))
    assert guard.main() == 1
    assert capsys.readouterr().out == ""


# --- the decision line, the log and the audit's reading of it ------------------------------------


def test_every_decision_is_one_json_line_and_exit_0(tree: Tree) -> None:
    for tool_input, expected in (({"file_path": str(tree.exchange / "req" / "a")}, "allow"),
                                 ({"file_path": "/etc/passwd"}, "deny")):
        code, out, err = tree.run(tree.call("dm-labeler", "Read", tool_input))
        assert (code, err) == (0, "") and out.endswith("\n") and out.count("\n") == 1
        assert json.loads(out)["hookSpecificOutput"]["permissionDecision"] == expected


def test_the_audit_reads_an_allow_as_a_guard_entry_and_a_deny_as_none(tree: Tree) -> None:
    command = GUARD_COMMAND
    _, allowed, _ = tree.run(tree.call("dm-labeler", "Read",
                                       {"file_path": str(tree.exchange / "req" / "a")}))
    _, denied, _ = tree.run(tree.call("dm-labeler", "Read", {"file_path": "/etc/passwd"}))
    entry = {"hookEvent": "PreToolUse", "command": command}
    assert audit._is_guard_entry({**entry, "stdout": allowed}, command) is True
    assert audit._is_guard_entry({**entry, "stdout": denied}, command) is False
    assert audit._is_guard_entry({**entry, "stdout": allowed}, FIXTURE_GUARD) is False


def test_each_call_is_logged_with_agent_type_and_tool_use_id_and_no_path(tree: Tree) -> None:
    secret = "/etc/passwd-marker"
    tree.run(tree.call("dm-labeler", "Read", {"file_path": secret}, tool_use_id="toolu_log1"))
    tree.run(tree.call("dm-driver", "Agent", agent_input(tree), tool_use_id="toolu_log2"))
    lines = [json.loads(line) for line in tree.log.read_text("utf-8").splitlines()]
    assert [(r["agent_type"], r["tool_name"], r["tool_use_id"], r["decision"]) for r in lines] == [
        ("dm-labeler", "Read", "toolu_log1", "deny"), ("dm-driver", "Agent", "toolu_log2", "allow")]
    assert all(r["event"] == "PreToolUse" and r["at"].endswith("Z") for r in lines)
    text = tree.log.read_text("utf-8")
    assert secret not in text and tree.serve.opaque not in text


def test_a_failure_logs_nothing(tree: Tree) -> None:
    tree.run(tree.call(_MISSING, "Read", {"file_path": "x"}))
    assert not tree.log.exists()


def test_the_installed_command_string_runs_the_guard_from_the_project_dir(tree: Tree) -> None:
    """The settings' command through a shell with `CLAUDE_PROJECT_DIR` set, as Claude Code does."""
    command = tree.made.camp.state().expectations["guardCommand"]
    payload = tree.call("dm-labeler", "Read", {"file_path": str(tree.exchange / "req" / "a")})
    done = subprocess.run(command, shell=True, input=json.dumps(payload), capture_output=True,
                          text=True, cwd=tree.exchange.parent,
                          env={**os.environ, "CLAUDE_PROJECT_DIR": str(tree.exchange)})
    assert done.returncode == 0
    assert json.loads(done.stdout)["hookSpecificOutput"]["permissionDecision"] == "allow"


def test_the_guard_imports_only_the_standard_library() -> None:
    import ast
    tree_ = ast.parse(GUARD.read_text("utf-8"))
    names = {alias.name.split(".")[0] for node in ast.walk(tree_) if isinstance(node, ast.Import)
             for alias in node.names}
    names |= {node.module.split(".")[0] for node in ast.walk(tree_)
              if isinstance(node, ast.ImportFrom) and node.module}
    assert names <= set(sys.stdlib_module_names) and "decisionmodels" not in names



def test_each_role_is_checked_against_its_own_template(tmp_path: Path) -> None:
    """The shipped templates are byte-identical, so this installs a copy where they differ."""
    from test_install import agents_copy, fresh_campaign
    made = fresh_campaign(tmp_path)
    copy = agents_copy(tmp_path, **{
        "spawn-generator.txt": lambda text: "Generate. " + text,
        "spawn-labeler.txt": lambda text: "Label. " + text})
    made.install(agents_dir=copy)
    exchange = made.camp.exchange_dir
    here = Tree(made, exchange, None)         # type: ignore[arg-type]
    request = exchange / "req" / "r1.jsonl"
    templates = {agent: (copy / f"spawn-{agent.removeprefix('dm-')}.txt").read_text("utf-8")
                 for agent in ("dm-generator", "dm-labeler")}
    assert templates["dm-generator"] != templates["dm-labeler"]
    for role, agent, other in (("generate", "dm-generator", "dm-labeler"),
                               ("label", "dm-labeler", "dm-generator")):
        request.write_text(json.dumps({"role": role}) + "\n", "utf-8")
        own = templates[agent].replace("{request_path}", str(request))
        theirs = templates[other].replace("{request_path}", str(request))
        assert here.decision(here.call("dm-driver", "Agent", {
            "subagent_type": agent, "prompt": own})) == "allow", agent
        assert here.decision(here.call("dm-driver", "Agent", {
            "subagent_type": agent, "prompt": theirs})) == "deny", agent
