"""The transcript audit (typed-decision-models U5.5; plan `docs/plans/typed-decision-models6.md`
§3.3, §3.4 and §5 U5.5 over plan4 `docs/plans/typed-decision-models4.md` §3.3, §3.7 and §5 U5.5).

`tests/fixtures/transcripts/` is hand-built in the shapes the probe report records
(`docs/test-reports/typed-decision-models-harness-probe-report.md`). The clean label run is H1's
foreground shape (§9.6): `driver.jsonl`'s `Agent` `tool_use`, its guard entry and `completed`
result, then `agent-a5c1e8d2b7f40936e.*`. A decoy run (`agent-a0f3e1d7c5b3a1f0e.*`) under another
`toolUseId` sorts first and breaks every pin, so a locator that picks it by anything but
`toolUseId` turns the clean run red. `guard-decisions.jsonl` holds the `tool_result` entries of
U6.7's D1, D2 and D3 (§10.3) and attempt 1's pre-hook validation error (§7), in their recorded key
sets with paths and ids replaced. Each test edits a copy of the clean run to match one row and
asserts every finding, not only the chosen one.
"""


from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from decisionmodels import audit, ingest

FIXTURES = Path(__file__).parent / "fixtures" / "transcripts"
SESSION = "5e551011-0000-4000-8000-0000000000d1"
AGENT = "a5c1e8d2b7f40936e"
DECOY = "a0f3e1d7c5b3a1f0e"
SPAWN_ID = "toolu_spawn0001"
REQ = "/campaign/exchange/req/q7m2x4k9p3r8t6v5.jsonl"
ANS = "/campaign/exchange/ans/q7m2x4k9p3r8t6v5.jsonl"

GUARD = "python3 guard_hook.py"

#: plan4 §3.4's spawn template; U5.7 owns the installed text.
SPAWN_TEMPLATE = (
    "Read {request_path} with offset 1, limit 1. Its `pages` field lists [offset, limit] pairs: "
    "read each remaining page with exactly that offset and limit. Follow the instructions on "
    "line 2, write the answer where the header says, then reply `done`."
)

EXPECTED = audit.Expected(
    agent_type="dm-labeler",
    model="claude-sonnet-5-5",
    effort="high",
    spawn_template=SPAWN_TEMPLATE,
    request_path=REQ,
    answer_path=ANS,
    guard_command=GUARD,
)

ACCEPT_HOOK = {"decision": "accept", "source": "hook", "reasonType": "hook"}
ALLOW = {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow",
                                "permissionDecisionReason": "ok"}}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


@dataclass
class Run:
    """An editable copy of the clean fixture run."""

    driver: list[dict[str, Any]]
    agent: list[dict[str, Any]] | None
    meta: dict[str, Any] | None
    extra: dict[str, str] = field(default_factory=dict)

    def stage(self, root: Path) -> Path:
        project = root / "project"
        subagents = project / SESSION / "subagents"
        subagents.mkdir(parents=True)
        for name in (f"agent-{DECOY}.jsonl", f"agent-{DECOY}.meta.json"):
            (subagents / name).write_bytes((FIXTURES / name).read_bytes())
        if self.agent is not None:
            write_jsonl(subagents / f"agent-{AGENT}.jsonl", self.agent)
        if self.meta is not None:
            (subagents / f"agent-{AGENT}.meta.json").write_text(json.dumps(self.meta), "utf-8")
        for name, text in self.extra.items():
            (subagents / name).write_text(text, "utf-8")
        driver = project / f"{SESSION}.jsonl"
        write_jsonl(driver, self.driver)
        return driver

    # -- views over the agent transcript ---------------------------------------------------------

    def assistant(self, message_id: str) -> list[dict[str, Any]]:
        assert self.agent is not None
        return [e for e in self.agent if e.get("type") == "assistant"
                and e["message"]["id"] == message_id]

    def tool_use(self, tool_use_id: str) -> dict[str, Any]:
        assert self.agent is not None
        for entry in self.agent:
            if entry.get("type") == "assistant":
                for block in entry["message"]["content"]:
                    if block.get("id") == tool_use_id:
                        return block
        raise KeyError(tool_use_id)

    def spawn_input(self) -> dict[str, Any]:
        return self.driver[1]["message"]["content"][0]["input"]

    def append(self, *entries: dict[str, Any]) -> None:
        assert self.agent is not None
        self.agent.extend(entries)


def write_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries), "utf-8")


def clean_run() -> Run:
    return Run(
        driver=read_jsonl(FIXTURES / "driver.jsonl"),
        agent=read_jsonl(FIXTURES / f"agent-{AGENT}.jsonl"),
        meta=json.loads((FIXTURES / f"agent-{AGENT}.meta.json").read_text("utf-8")),
    )


#: plan6 §3.4's `launchFailBound` default, as `init` records it.
BOUND = timedelta(hours=6)
LAUNCH = audit.Launch(fail_bound=BOUND)


def audit_of(tmp_path: Path, run: Run, *, expected: audit.Expected = EXPECTED,
             earlier: int = 0, launch: audit.Launch = LAUNCH, **kwargs: Any) -> audit.Audit:
    return audit.audit_run(run.stage(tmp_path), expected, earlier_no_attempt=earlier,
                           launch=launch, **kwargs)


def found(result: audit.Audit) -> list[tuple[str, str]]:
    return [(f.void_class, f.reason) for f in result.findings]


def chosen(result: audit.Audit) -> tuple[str, str | None, str | None]:
    return result.action, result.void_class, result.reason


# -- entry builders, in the shapes of plan4 §2 -----------------------------------------------

def _envelope(**fields: Any) -> dict[str, Any]:
    return {"isSidechain": True, "agentId": AGENT, "cwd": "/campaign/exchange",
            "sessionId": SESSION, "version": "2.1.316", **fields}


def attachment(kind: str, **fields: Any) -> dict[str, Any]:
    return _envelope(type="attachment", attachment={"type": kind, **fields})


def synthetic(error: str | None, **fields: Any) -> dict[str, Any]:
    entry = _envelope(**fields, type="assistant", isApiErrorMessage=error is not None, message={
        "id": "9b1f0a52-0000-4000-8000-000000000001", "model": "<synthetic>", "role": "assistant",
        "type": "message", "stop_reason": "stop_sequence", "content": [{"type": "text",
                                                                         "text": "…"}],
        "usage": {"input_tokens": 0, "cache_creation_input_tokens": 0,
                  "cache_read_input_tokens": 0, "output_tokens": 0}})
    if error is not None:
        entry["error"] = error
    return entry


def tool_use(tool_use_id: str, name: str, tool_input: dict[str, Any], *,
             message_id: str = "msg_x0001") -> dict[str, Any]:
    return _envelope(type="assistant", message={
        "model": "claude-sonnet-5-5", "id": message_id, "type": "message", "role": "assistant",
        "content": [{"type": "tool_use", "id": tool_use_id, "name": name, "input": tool_input}],
        "stop_reason": "tool_use", "usage": {"input_tokens": 1, "cache_creation_input_tokens": 0,
                                             "cache_read_input_tokens": 0, "output_tokens": 1}})


def guard_entry(tool_use_id: str, name: str, *, command: str = GUARD,
                stdout: str = json.dumps(ALLOW) + "\n") -> dict[str, Any]:
    """The guard's `hook_success` for an allowed call (report §9.6, §10.3 rows 1, 2 and 6)."""
    return attachment("hook_success", hookName=f"PreToolUse:{name}", toolUseID=tool_use_id,
                      hookEvent="PreToolUse", content="", stdout=stdout, stderr="", exitCode=0,
                      command=command, durationMs=40)


def recorded(tool_use_id: str) -> dict[str, Any]:
    """A `tool_result` entry of `guard-decisions.jsonl`, in the key sets the probes recorded:
    U6.7's D1 (`toolu_g0001`, a guard deny), D2 (`toolu_g0002`, a guard failure under
    `onFailure: "block"`) and D3 (`toolu_g0003`, no decision; `acceptEdits` decided), report
    §10.3; and attempt 1's pre-hook `InputValidationError` (`toolu_v0001`), report §7."""
    for entry in read_jsonl(FIXTURES / "guard-decisions.jsonl"):
        if entry["message"]["content"][0]["tool_use_id"] == tool_use_id:
            return entry
    raise KeyError(tool_use_id)


def tool_call(tool_use_id: str, name: str, tool_input: dict[str, Any], *, guard: bool = True,
              message_id: str = "msg_x0001") -> list[dict]:
    """An assistant `tool_use`, the guard's allowing `hook_success` (none when `guard` is false)
    and its `accept`/`hook` `tool_result` (H6's allowed shape, report §9.4)."""
    entries = [tool_use(tool_use_id, name, tool_input, message_id=message_id)]
    if guard:
        entries.append(guard_entry(tool_use_id, name))
    entries.append(_envelope(type="user", permissionDecision=dict(ACCEPT_HOOK), message={
        "role": "user", "content": [
            {"tool_use_id": tool_use_id, "type": "tool_result", "content": "x"}]}))
    return entries


def invalid_call() -> list[dict[str, Any]]:
    """A `Write` refused by input validation before any hook ran (a counted row)."""
    return [tool_use("toolu_v0001", "Write", {"file_path": ANS}), recorded("toolu_v0001")]


def prompt_only(run: Run) -> None:
    """A driver transcript holding the `-p` prompt and nothing after it."""
    run.driver = [run.driver[0]]


def refusal(**fields: Any) -> dict[str, Any]:
    """F7's assistant entry: `stop_reason: "refusal"` (report §7, the REFUSAL entry)."""
    return _envelope(**fields, type="assistant", message={
        "model": "claude-sonnet-5-5", "id": "msg_ref0001", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": "…"}], "stop_reason": "refusal",
        "usage": {"input_tokens": 1, "cache_creation_input_tokens": 0,
                  "cache_read_input_tokens": 0, "output_tokens": 0}})


def refused_notice(**fields: Any) -> dict[str, Any]:
    """F7's `user` entry: `isMeta: true` and a `refusedMessageId` key (report §7)."""
    return _envelope(**fields, type="user", isMeta=True, refusedMessageId="msg_ref0001",
                     promptId="p0000001", turnCompanion=True,
                     message={"role": "user", "content": "…"})


DRIVER_SIDE = {"isSidechain": False, "agentId": None}


def drop_tool_result(run: Run) -> None:
    """H3a's kill shape (report §9.6): the driver ends at its `Agent` `tool_use`, with no
    `hook_success`, no `tool_result` and no reply."""
    run.driver = run.driver[:3]


def spawn_result(run: Run) -> dict[str, Any]:
    """The driver's `tool_result` entry for the spawn."""
    return next(e for e in run.driver if e["type"] == "user"
                and isinstance(e["message"]["content"], list)
                and e["message"]["content"][0].get("tool_use_id") == SPAWN_ID)


def driver_text(content: str, **fields: Any) -> dict[str, Any]:
    """A driver `user` entry with string content (report §9.6, H2's three string entries)."""
    return {"parentUuid": None, "isSidechain": False, "type": "user",
            "message": {"role": "user", "content": content}, "cwd": "/campaign/exchange",
            "sessionId": SESSION, "version": "2.1.316", **fields}


def stopped_notification() -> dict[str, Any]:
    """F5's `stopped` `<task-notification>`: no `<tool-use-id>` (report §7, session fb3a881a)."""
    return driver_text(f"<task-notification>\n<task-id>{AGENT}</task-id>\n<status>stopped</status>"
                       "\n<summary>stopped</summary>\n<note>…</note>\n</task-notification>",
                       origin={"kind": "task-notification"})


def background(run: Run, *, notified: bool) -> None:
    """F1/H2's shape: an `async_launched` result and a `background` meta; with `notified`, H2's
    second driver turn (an `isMeta` entry, the `<task-notification>`, a second reply)."""
    result = spawn_result(run)
    prompt = result["toolUseResult"]["prompt"]
    result["toolUseResult"] = {
        "agentId": AGENT, "canContinueAgent": False, "canReadOutputFile": False,
        "description": "label batch", "isAsync": True, "outputFile": "/tmp/out", "prompt": prompt,
        "resolvedModel": "claude-sonnet-5-5", "status": "async_launched"}
    assert run.meta is not None
    run.meta.update(requestShape="background", launchedWithDefaults=True)
    if notified:
        reply = copy.deepcopy(run.driver[-1])
        run.driver += [
            driver_text("<system-reminder>…</system-reminder>", isMeta=True,
                        queueSkipAttachments=True, turnCompanion=True),
            driver_text(f"<task-notification>\n<task-id>{AGENT}</task-id>\n<tool-use-id>{SPAWN_ID}"
                        "</tool-use-id>\n<status>completed</status>\n</task-notification>",
                        origin={"kind": "task-notification", "producer": "session-task"}),
            reply]


def insert_before_reply(run: Run, *entries: dict[str, Any]) -> None:
    """Insert entries before the final api-request and `done` reply."""
    assert run.agent is not None
    run.agent[-2:-2] = list(entries)


# == Location ================================================================================

def test_the_run_is_found_through_the_agent_tool_use_and_meta_tool_use_id(tmp_path: Path) -> None:
    result = audit_of(tmp_path, clean_run())
    assert chosen(result) == (audit.CLEAN, None, None)
    assert result.findings == ()
    assert result.facts is not None and result.facts.agent_id == AGENT


def test_the_api_request_blob_replay_of_the_spawn_is_not_a_second_tool_use(
    tmp_path: Path,
) -> None:
    run = clean_run()
    assert run.driver[2]["type"] == "api-request-blob"
    assert audit_of(tmp_path, run).action == audit.CLEAN


def test_a_spawn_prompt_off_by_one_byte_is_found_then_halts_as_a_driver_void(
    tmp_path: Path,
) -> None:
    run = clean_run()
    off = SPAWN_TEMPLATE.replace("{request_path}", REQ) + "."
    run.spawn_input()["prompt"] = off
    assert run.agent is not None
    run.agent[0]["message"]["content"] = off
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.HALT, audit.DRIVER, "spawnPromptMismatch")
    assert found(result) == [(audit.DRIVER, "spawnPromptMismatch")]
    assert result.facts is not None and result.facts.agent_id == AGENT


def test_a_first_user_message_differing_from_an_exact_spawn_prompt_halts(tmp_path: Path) -> None:
    run = clean_run()
    assert run.agent is not None
    run.agent[0]["message"]["content"] = run.agent[0]["message"]["content"].replace("done", "ok")
    assert found(audit_of(tmp_path, run)) == [(audit.DRIVER, "spawnPromptMismatch")]


def test_with_no_tool_use_the_batch_is_reissued_with_no_attempt(tmp_path: Path) -> None:
    run = clean_run()
    run.driver = [run.driver[0]]
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.REISSUE, audit.NEVER_SPAWNED, "neverSpawned")
    assert result.facts is None


def test_a_tool_use_for_another_request_is_not_this_requests_spawn(tmp_path: Path) -> None:
    run = clean_run()
    run.spawn_input()["prompt"] = SPAWN_TEMPLATE.replace("{request_path}", REQ.replace("q7", "x7"))
    run.driver = run.driver[:2] + run.driver[3:]
    assert chosen(audit_of(tmp_path, run))[2] == "neverSpawned"


def test_a_second_never_spawned_outcome_for_one_batch_halts(tmp_path: Path) -> None:
    run = clean_run()
    run.driver = [run.driver[0]]
    result = audit_of(tmp_path, run, earlier=1)
    assert chosen(result) == (audit.HALT, audit.DRIVER, "neverSpawned")


def test_a_tool_use_with_no_subagent_transcript_halts(tmp_path: Path) -> None:
    run = clean_run()
    run.agent, run.meta = None, None
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.HALT, audit.DRIVER, "transcriptMissing")
    assert result.facts is None


def test_a_meta_file_with_no_transcript_beside_it_halts(tmp_path: Path) -> None:
    run = clean_run()
    run.agent = None
    assert chosen(audit_of(tmp_path, run)) == (audit.HALT, audit.DRIVER, "transcriptMissing")


def test_an_empty_subagent_transcript_is_audited_not_skipped(tmp_path: Path) -> None:
    run = clean_run()
    run.agent = []
    result = audit_of(tmp_path, run)
    assert result.action == audit.HALT
    assert (audit.DRIVER, "spawnPromptMismatch") in found(result)


def test_two_transcripts_for_one_tool_use_halt(tmp_path: Path) -> None:
    run = clean_run()
    twin = "a7777777777777777"
    run.extra = {
        f"agent-{twin}.meta.json": json.dumps(run.meta),
        f"agent-{twin}.jsonl": (FIXTURES / f"agent-{AGENT}.jsonl").read_text("utf-8"),
    }
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.HALT, audit.DRIVER, "transcriptAmbiguous")
    assert result.facts is None


def test_two_spawns_of_one_request_halt(tmp_path: Path) -> None:
    run = clean_run()
    again = copy.deepcopy(run.driver[1])
    again["message"]["content"][0]["id"] = "toolu_spawn0002"
    run.driver.append(again)
    assert chosen(audit_of(tmp_path, run)) == (audit.HALT, audit.DRIVER, "toolUseRepeated")


# == Incomplete and the no-attempt cap =======================================================

def test_no_tool_result_is_incomplete_with_no_attempt(tmp_path: Path) -> None:
    run = clean_run()
    drop_tool_result(run)
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.REISSUE, audit.INCOMPLETE, "incomplete")
    assert found(result) == [(audit.INCOMPLETE, "incomplete")]


def test_a_second_incomplete_outcome_for_one_batch_halts(tmp_path: Path) -> None:
    run = clean_run()
    drop_tool_result(run)
    assert chosen(audit_of(tmp_path, run, earlier=1)) == (audit.HALT, audit.DRIVER, "incomplete")


def test_a_second_incomplete_named_by_an_interrupted_event_does_not_halt(tmp_path: Path) -> None:
    run = clean_run()
    drop_tool_result(run)
    assert chosen(audit_of(tmp_path, run, earlier=1, interrupted=True)) == (
        audit.REISSUE, audit.INCOMPLETE, "incomplete")


def test_the_no_attempt_cap_is_shared_by_never_spawned_and_incomplete(tmp_path: Path) -> None:
    """plan4 §5 U5.6: "the second never-spawned or incomplete outcome for one batch halts"; the
    caller counts both kinds into `earlier_no_attempt`."""
    run = clean_run()
    run.driver = [run.driver[0]]
    assert audit_of(tmp_path / "a", run, earlier=0).action == audit.REISSUE
    assert audit_of(tmp_path / "b", run, earlier=2).action == audit.HALT


# == Completion (plan6 §3.4) =================================================================

def test_the_foreground_shape_with_an_end_turn_reply_is_clean(tmp_path: Path) -> None:
    """F2/H1: `completed`, `requestShape: "foreground"`, `finished: true`."""
    run = clean_run()
    assert spawn_result(run)["toolUseResult"]["status"] == "completed"
    assert run.meta is not None and run.meta["requestShape"] == "foreground"
    assert chosen(audit_of(tmp_path, run)) == (audit.CLEAN, None, None)


@pytest.mark.parametrize("notification", [False, True])
def test_the_kill_shape_is_incomplete(tmp_path: Path, notification: bool) -> None:
    """H3a: the driver ends at its `Agent` `tool_use` (no guard entry, no result), the meta has
    no `finished`; also with F5's `stopped` notification from a restart."""
    run = clean_run()
    drop_tool_result(run)
    assert run.meta is not None and run.meta["requestShape"] == "foreground"
    del run.meta["finished"]
    if notification:
        run.driver.append(stopped_notification())
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.REISSUE, audit.INCOMPLETE, "incomplete")
    assert found(result) == [(audit.INCOMPLETE, "incomplete")]


def test_a_killed_subagents_last_tool_use_with_no_guard_entry_is_in_flight(tmp_path: Path) -> None:
    """The hook ran, but a kill before the result loses its entry (H3a, gate (c))."""
    run = clean_run()
    drop_tool_result(run)
    assert run.agent is not None
    run.agent = run.agent[:15]
    assert run.agent[-1]["message"]["content"][0]["id"] == "toolu_w0001"
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.REISSUE, audit.INCOMPLETE, "incomplete")
    assert found(result) == [(audit.INCOMPLETE, "incomplete")]


def test_parallel_calls_killed_before_their_attachments_are_in_flight(tmp_path: Path) -> None:
    """Parallel calls persist as `tool_use`, `tool_use`, then entry and result pairs (P3-m1)."""
    run = clean_run()
    drop_tool_result(run)
    assert run.agent is not None
    run.agent = run.agent[:15] + [tool_use("toolu_r0009", "Read", {"file_path": REQ},
                                           message_id="msg_s0003")]
    assert found(audit_of(tmp_path, run)) == [(audit.INCOMPLETE, "incomplete")]


def test_an_agent_tool_use_with_no_subagent_transcript_is_in_flight(tmp_path: Path) -> None:
    run = clean_run()
    drop_tool_result(run)
    run.agent, run.meta = None, None
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.REISSUE, audit.INCOMPLETE, "incomplete")
    assert found(result) == [(audit.INCOMPLETE, "incomplete")]


def test_in_a_completed_run_a_tool_use_with_no_result_is_hook_missing(tmp_path: Path) -> None:
    """Nothing is in flight once the driver holds the spawn's result."""
    run = clean_run()
    insert_before_reply(run, tool_use("toolu_r0009", "Read", {"file_path": REQ}))
    assert found(audit_of(tmp_path, run)) == [(audit.ENVIRONMENTAL, "hookMissing")]


def test_the_drivers_agent_call_with_no_guard_entry_is_hook_missing(tmp_path: Path) -> None:
    run = clean_run()
    run.driver = [e for e in run.driver if e["type"] != "attachment"]
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.HALT, audit.ENVIRONMENTAL, "hookMissing")


def test_a_foreign_subagent_stop_entry_changes_nothing(tmp_path: Path) -> None:
    """F6's foreign `SubagentStop` with the driver's `agent_type`: no row reads that event, so it
    neither voids a clean run nor stands in for the driver's missing guard entry."""
    foreign = {**guard_entry(SPAWN_ID, "Agent"), "isSidechain": False}
    foreign["attachment"].update(hookEvent="SubagentStop", hookName="SubagentStop",
                                 agent_type="dm-driver")
    run = clean_run()
    run.driver.append(foreign)
    assert chosen(audit_of(tmp_path / "clean", run)) == (audit.CLEAN, None, None)
    unguarded = clean_run()
    unguarded.driver[3] = foreign
    assert chosen(audit_of(tmp_path / "unguarded", unguarded))[2] == "hookMissing"


@pytest.mark.parametrize("value", [True, False])
def test_request_non_interactive_changes_no_class(tmp_path: Path, value: bool) -> None:
    run = clean_run()
    assert run.meta is not None
    run.meta["requestNonInteractive"] = value
    assert chosen(audit_of(tmp_path / "clean", run)) == (audit.CLEAN, None, None)
    drop_tool_result(run)
    assert chosen(audit_of(tmp_path / "killed", run))[2] == "incomplete"


@pytest.mark.parametrize("notified", [False, True])
def test_a_background_spawn_with_a_complete_answer_halts(tmp_path: Path, notified: bool) -> None:
    """F1/H2: `-p` waits for the background run, whose answer is complete; it halts all the same."""
    run = clean_run()
    background(run, notified=notified)
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.HALT, audit.ENVIRONMENTAL, "backgroundSpawn")
    assert found(result) == [(audit.ENVIRONMENTAL, "backgroundSpawn")]


def test_an_async_launched_result_alone_halts_as_a_background_spawn(tmp_path: Path) -> None:
    run = clean_run()
    spawn_result(run)["toolUseResult"]["status"] = "async_launched"
    assert chosen(audit_of(tmp_path, run))[2] == "backgroundSpawn"


def test_a_background_meta_alone_halts_as_a_background_spawn(tmp_path: Path) -> None:
    run = clean_run()
    assert run.meta is not None
    run.meta["requestShape"] = "background"
    assert chosen(audit_of(tmp_path, run))[2] == "backgroundSpawn"


@pytest.mark.parametrize("edit", [
    lambda run: run.meta.pop("finished"),
    lambda run: run.meta.update(finished=False),
    lambda run: spawn_result(run)["toolUseResult"].update(status="paused"),
    lambda run: spawn_result(run)["toolUseResult"].pop("status"),
    lambda run: run.meta.update(requestShape="detached"),
    lambda run: run.meta.pop("requestShape"),
])
def test_an_unknown_completion_shape_halts(tmp_path: Path, edit) -> None:
    run = clean_run()
    edit(run)
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.HALT, audit.ENVIRONMENTAL, "spawnShapeUnknown")
    assert found(result) == [(audit.ENVIRONMENTAL, "spawnShapeUnknown")]


@pytest.mark.parametrize("edit", [
    lambda meta: meta.update(requestShape="detached"),
    lambda meta: meta.pop("requestShape"),
])
def test_an_in_flight_run_whose_meta_is_not_foreground_halts(tmp_path: Path, edit) -> None:
    """P9-F1: the meta row has no completion condition, and the meta is written at spawn, so
    H3a's killed run (`requestShape: "foreground"`, no `finished`) still re-issues."""
    run = clean_run()
    drop_tool_result(run)
    assert run.meta is not None
    del run.meta["finished"]
    edit(run.meta)
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.HALT, audit.ENVIRONMENTAL, "spawnShapeUnknown")
    assert found(result) == [(audit.ENVIRONMENTAL, "spawnShapeUnknown"),
                             (audit.INCOMPLETE, "incomplete")]


def test_the_spawns_result_is_matched_by_its_tool_use_id(tmp_path: Path) -> None:
    """P8-F3 (P1): a driver `tool_result` for another id is not the spawn's."""
    run = clean_run()
    other = copy.deepcopy(spawn_result(run))
    other["message"]["content"][0]["tool_use_id"] = "toolu_other0099"
    drop_tool_result(run)
    run.driver.append(other)
    assert chosen(audit_of(tmp_path, run)) == (audit.REISSUE, audit.INCOMPLETE, "incomplete")


# == Safeguards (plan6 §3.4, §3.5) ===========================================================

def test_a_refusal_then_a_malformed_partial_write_halts_and_ingests_nothing(
    tmp_path: Path,
) -> None:
    """Not a counted `malformed`: the safeguards row comes before every counted row."""
    run = clean_run()
    assert run.agent is not None
    run.agent = run.agent[:12] + [
        refusal(), *tool_call("toolu_w0009", "Write", {"file_path": ANS, "content": "{\"bodyS"})]
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.HALT, audit.ENVIRONMENTAL, "safeguardStop")
    assert found(result) == [(audit.ENVIRONMENTAL, "safeguardStop"),
                             (audit.RUN_ATTRIBUTABLE, "maxTurns")]
    assert not result.ingests_answer


@pytest.mark.parametrize("where", ["driver", "agent"])
def test_a_refused_message_notice_alone_halts(tmp_path: Path, where: str) -> None:
    run = clean_run()
    if where == "driver":
        run.driver.append(refused_notice(**DRIVER_SIDE))
    else:
        insert_before_reply(run, refused_notice())
    assert found(audit_of(tmp_path, run)) == [(audit.ENVIRONMENTAL, "safeguardStop")]


@pytest.mark.parametrize("where", ["driver", "agent"])
def test_a_refusal_alone_halts(tmp_path: Path, where: str) -> None:
    run = clean_run()
    if where == "driver":
        run.driver.append(refusal(**DRIVER_SIDE))
    else:
        insert_before_reply(run, refusal())
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.HALT, audit.ENVIRONMENTAL, "safeguardStop")


def test_a_refusal_beats_a_usage_stop(tmp_path: Path) -> None:
    run = clean_run()
    run.append(synthetic("rate_limit"))
    run.driver.append(refusal(**DRIVER_SIDE))
    assert chosen(audit_of(tmp_path, run))[2] == "safeguardStop"


# == The driver session (plan6 §3.3, §3.4) ====================================================

def test_a_driver_on_another_model_halts(tmp_path: Path) -> None:
    run = clean_run()
    run.driver[1]["message"]["model"] = "claude-opus-5-5"
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.HALT, audit.ENVIRONMENTAL, "driverModelMismatch")
    assert found(result) == [(audit.ENVIRONMENTAL, "driverModelMismatch")]


def test_a_driver_on_another_model_halts_even_before_any_agent_call(tmp_path: Path) -> None:
    run = clean_run()
    run.driver[1]["message"]["content"] = [{"type": "text", "text": "ok"}]
    run.driver[1]["message"]["model"] = "claude-opus-5-5"
    run.driver = run.driver[:2]
    assert chosen(audit_of(tmp_path, run))[2] == "driverModelMismatch"


def test_a_driver_synthetic_error_halts(tmp_path: Path) -> None:
    run = clean_run()
    run.driver.append(synthetic("authentication_failed", **DRIVER_SIDE))
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.HALT, audit.ENVIRONMENTAL, "syntheticError")


def test_a_driver_rate_limit_with_no_agent_call_waits_uncapped(tmp_path: Path) -> None:
    """Ten in a row for one batch, and none halts: a usage stop is never capped."""
    run = clean_run()
    prompt_only(run)
    run.driver.append(synthetic("rate_limit", **DRIVER_SIDE))
    for earlier in range(10):
        result = audit_of(tmp_path / str(earlier), run, earlier=earlier)
        assert chosen(result) == (audit.WAIT, audit.USAGE_STOP, "rateLimit")
        assert found(result) == [(audit.USAGE_STOP, "rateLimit")]


def test_a_driver_rate_limit_after_the_agent_call_waits(tmp_path: Path) -> None:
    run = clean_run()
    drop_tool_result(run)
    run.driver.append(synthetic("rate_limit", **DRIVER_SIDE))
    assert found(audit_of(tmp_path, run)) == [(audit.USAGE_STOP, "rateLimit")]


# == Launch (plan6 §3.4 launchFailed) ========================================================

T0 = datetime(2026, 11, 1, 10, 0, tzinfo=UTC)


def launch_failed(tmp_path: Path, *, earlier: int = 0, **launch: Any) -> audit.Audit:
    missing = tmp_path / "project" / f"{SESSION}.jsonl"
    return audit.audit_run(missing, EXPECTED, earlier_no_attempt=earlier,
                           launch=audit.Launch(**{"fail_bound": BOUND, **launch}))


def test_no_transcript_is_launch_failed_uncapped(tmp_path: Path) -> None:
    for earlier in (0, 1, 5):
        result = launch_failed(tmp_path, earlier=earlier, exit_code=1, at=T0)
        assert chosen(result) == (audit.REISSUE, audit.LAUNCH_FAILED, "launchFailed")
        assert result.facts is None


@pytest.mark.parametrize("after, reason", [
    (timedelta(hours=6, minutes=1), "launchFailing"),
    (timedelta(hours=6, seconds=1), "launchFailing"),
    (timedelta(hours=6), "launchFailed"),
    (timedelta(hours=5, minutes=59), "launchFailed"),
])
def test_launches_producing_nothing_past_the_bound_halt(
    tmp_path: Path, after: timedelta, reason: str
) -> None:
    result = launch_failed(tmp_path, exit_code=1, at=T0 + after, earlier_failures=(T0,),
                           fail_bound=timedelta(hours=6))
    assert chosen(result)[2] == reason
    assert chosen(result)[0] == (audit.HALT if reason == "launchFailing" else audit.REISSUE)


def test_a_launch_that_produced_something_resets_the_clock(tmp_path: Path) -> None:
    result = launch_failed(tmp_path, exit_code=1, at=T0 + timedelta(hours=6, minutes=1),
                           earlier_failures=(T0, T0 + timedelta(hours=2)),
                           last_produced=T0 + timedelta(hours=1), fail_bound=timedelta(hours=6))
    assert chosen(result)[2] == "launchFailed"


@pytest.mark.parametrize("exit_code, outcome", [
    (1, (audit.REISSUE, audit.LAUNCH_FAILED, "launchFailed")),
    (0, (audit.REISSUE, audit.NEVER_SPAWNED, "neverSpawned")),
])
def test_a_prompt_only_transcript_is_launch_failed_or_never_spawned_by_exit_code(
    tmp_path: Path, exit_code: int, outcome: tuple
) -> None:
    run = clean_run()
    prompt_only(run)
    launch = audit.Launch(exit_code=exit_code, at=T0, fail_bound=BOUND)
    result = audit_of(tmp_path, run, launch=launch)
    assert chosen(result) == outcome


@pytest.mark.parametrize("exit_code, outcome", [
    (1, (audit.REISSUE, audit.LAUNCH_FAILED, "launchFailed")),
    (0, (audit.REISSUE, audit.NEVER_SPAWNED, "neverSpawned")),
])
def test_an_empty_driver_transcript_is_launch_failed_or_never_spawned_by_exit_code(
    tmp_path: Path, exit_code: int, outcome: tuple
) -> None:
    """P9-F4 (M17): an empty file is a transcript with no `Agent` call, not a missing one."""
    run = clean_run()
    run.driver = []
    launch = audit.Launch(exit_code=exit_code, at=T0, fail_bound=BOUND)
    result = audit_of(tmp_path, run, launch=launch)
    assert chosen(result) == outcome


def test_the_launch_fail_bound_is_required(tmp_path: Path) -> None:
    """P9-F4: the bound recorded at `init` is the caller's to pass; no default stands in for it."""
    with pytest.raises(TypeError, match="fail_bound"):
        audit.Launch(exit_code=1, at=T0)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="launch"):
        audit.audit_run(tmp_path / "missing.jsonl", EXPECTED,  # type: ignore[call-arg]
                        earlier_no_attempt=0)


def test_a_launch_failure_records_no_run_and_no_status(tmp_path: Path) -> None:
    result = launch_failed(tmp_path, exit_code=1, at=T0)
    verdict = audit.combine(result, None)
    assert verdict.status is None and verdict.reissues_under_new_name
    assert not verdict.counts_attempt and not verdict.halts


# == Driver rows =============================================================================

@pytest.mark.parametrize("key, value", [
    ("model", "claude-sonnet-5-5"), ("effort", "high"), ("isolation", "worktree"),
    ("run_in_background", False),
])
def test_an_override_key_on_the_tool_use_halts_as_a_driver_void(
    tmp_path: Path, key: str, value: Any
) -> None:
    run = clean_run()
    run.spawn_input()[key] = value
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.HALT, audit.DRIVER, "toolUseOverride")
    assert found(result) == [(audit.DRIVER, "toolUseOverride")]


def test_another_subagent_type_on_the_tool_use_halts_as_a_driver_void(tmp_path: Path) -> None:
    run = clean_run()
    run.spawn_input()["subagent_type"] = "dm-generator"
    assert found(audit_of(tmp_path, run)) == [(audit.DRIVER, "subagentTypeMismatch")]


def test_an_effort_key_on_the_tool_use_is_a_driver_void_not_an_environmental_one(
    tmp_path: Path,
) -> None:
    """The run's effort follows the `tool_use`'s `effort`; the effort row is environmental only
    when the `tool_use` carries none."""
    run = clean_run()
    run.spawn_input()["effort"] = "low"
    assert run.agent is not None
    for entry in run.agent:
        if entry["type"] == "api-request":
            entry["params"]["output_config"]["effort"] = "low"
    result = audit_of(tmp_path, run)
    assert found(result) == [(audit.DRIVER, "toolUseOverride")]


# == Environmental rows ======================================================================

def test_an_env_var_effort_is_environmental(tmp_path: Path) -> None:
    run = clean_run()
    assert run.agent is not None
    run.agent[3]["params"]["output_config"]["effort"] = "low"
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.HALT, audit.ENVIRONMENTAL, "effortOverride")
    assert found(result) == [(audit.ENVIRONMENTAL, "effortOverride")]


def test_a_run_with_no_recorded_effort_is_environmental(tmp_path: Path) -> None:
    run = clean_run()
    assert run.agent is not None
    run.agent = [e for e in run.agent if e["type"] != "api-request"]
    assert found(audit_of(tmp_path, run)) == [(audit.ENVIRONMENTAL, "effortOverride")]


@pytest.mark.parametrize("agent_type", ["fork", "dm-generator"])
def test_a_meta_agent_type_other_than_the_roles_is_environmental(
    tmp_path: Path, agent_type: str
) -> None:
    run = clean_run()
    assert run.meta is not None
    run.meta["agentType"] = agent_type
    assert found(audit_of(tmp_path, run)) == [(audit.ENVIRONMENTAL, "agentTypeMismatch")]


def test_a_message_model_other_than_the_pin_is_environmental(tmp_path: Path) -> None:
    run = clean_run()
    run.assistant("msg_s0003")[1]["message"]["model"] = "claude-sonnet-5-0"
    assert found(audit_of(tmp_path, run)) == [(audit.ENVIRONMENTAL, "modelMismatch")]


def test_an_instructions_file_outside_the_declared_set_is_environmental(tmp_path: Path) -> None:
    run = clean_run()
    insert_before_reply(run, attachment("instructions", files=[
        {"path": "/campaign/CLAUDE.md", "type": "Project", "content": "Be brief."}]))
    assert found(audit_of(tmp_path, run)) == [(audit.ENVIRONMENTAL, "instructionsMismatch")]


def test_instructions_match_the_declared_set_by_path_and_sha256(tmp_path: Path) -> None:
    run = clean_run()
    insert_before_reply(run, attachment("instructions", files=[
        {"path": "/campaign/CLAUDE.md", "type": "Project", "content": "Be brief."}]))
    sha = hashlib.sha256(b"Be brief.").hexdigest()
    declared = replace(EXPECTED, instructions=frozenset({("/campaign/CLAUDE.md", sha)}))
    assert audit_of(tmp_path / "same", run, expected=declared).action == audit.CLEAN
    edited = replace(EXPECTED, instructions=frozenset({("/campaign/CLAUDE.md", "0" * 64)}))
    assert found(audit_of(tmp_path / "sha", run, expected=edited)) == [
        (audit.ENVIRONMENTAL, "instructionsMismatch")]
    moved = replace(EXPECTED, instructions=frozenset({("/campaign/AGENTS.md", sha)}))
    assert found(audit_of(tmp_path / "path", run, expected=moved)) == [
        (audit.ENVIRONMENTAL, "instructionsMismatch")]


def test_a_declared_instructions_file_that_is_absent_is_environmental(tmp_path: Path) -> None:
    declared = replace(EXPECTED, instructions=frozenset({("/campaign/CLAUDE.md", "0" * 64)}))
    assert found(audit_of(tmp_path, clean_run(), expected=declared)) == [
        (audit.ENVIRONMENTAL, "instructionsMismatch")]


def test_an_mcp_instructions_delta_is_environmental(tmp_path: Path) -> None:
    run = clean_run()
    insert_before_reply(run, attachment("mcp_instructions_delta", addedNames=["x"],
                                        addedBlocks=["…"], removedNames=[]))
    assert found(audit_of(tmp_path, run)) == [(audit.ENVIRONMENTAL, "mcpInstructions")]


def test_an_mcp_name_in_a_deferred_tools_record_is_environmental(tmp_path: Path) -> None:
    run = clean_run()
    assert run.agent is not None
    run.agent[2]["attachment"]["entries"] = [{"name": "mcp__server__tool", "description": "…"}]
    assert found(audit_of(tmp_path, run)) == [(audit.ENVIRONMENTAL, "mcpDeferredTool")]


def test_a_deferred_tools_record_without_mcp_names_passes(tmp_path: Path) -> None:
    run = clean_run()
    assert run.agent is not None
    run.agent[2]["attachment"]["entries"] = [{"name": "WebFetch", "description": "…"}]
    assert audit_of(tmp_path, run).action == audit.CLEAN


def test_a_git_status_key_in_session_context_is_environmental(tmp_path: Path) -> None:
    run = clean_run()
    assert run.agent is not None
    run.agent[1]["attachment"]["context"]["gitStatus"] = "Current branch: main"
    assert found(audit_of(tmp_path, run)) == [(audit.ENVIRONMENTAL, "gitStatus")]


# == Guard decisions (plan6 §3.4) ===========================================================

def test_an_allowed_call_with_its_guard_entry_matches_no_row(tmp_path: Path) -> None:
    """H6's allowed shape: `accept`/`hook` on the `tool_result` and the guard's entry."""
    run = clean_run()
    insert_before_reply(run, *tool_call("toolu_p0001", "Read", {"file_path": REQ}))
    assert chosen(audit_of(tmp_path, run)) == (audit.CLEAN, None, None)


@pytest.mark.parametrize("tool_use_id", ["toolu_g0001", "toolu_g0002"])
def test_a_guard_deny_or_a_blocked_guard_failure_halts_as_guard_rejected(
    tmp_path: Path, tool_use_id: str
) -> None:
    """D1 and D2 are structurally identical (report §10.2): `is_error`, `reject`/`hook`,
    `toolDenialKind` and no attachment. Both halt as one class."""
    run = clean_run()
    insert_before_reply(run, tool_use(tool_use_id, "Write", {"file_path": ANS, "content": "ok"}),
                        recorded(tool_use_id))
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.HALT, audit.ENVIRONMENTAL, "guardRejected")
    assert found(result) == [(audit.ENVIRONMENTAL, "guardRejected")]


def test_a_call_the_permission_flow_decided_halts_as_hook_missing(tmp_path: Path) -> None:
    """D3: the guard printed nothing, so `acceptEdits` decided (`accept`/`config`)."""
    run = clean_run()
    insert_before_reply(run, tool_use("toolu_g0003", "Write", {"file_path": ANS, "content": "ok"}),
                        recorded("toolu_g0003"))
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.HALT, audit.ENVIRONMENTAL, "hookMissing")
    assert found(result) == [(audit.ENVIRONMENTAL, "hookMissing")]


def test_an_accept_by_hook_with_no_guard_entry_halts_as_hook_missing(tmp_path: Path) -> None:
    """Not a counted `answerMissing`: the answer `Write` itself lacks its guard entry."""
    run = clean_run()
    assert run.agent is not None
    run.agent = [e for e in run.agent if not (
        e["type"] == "attachment" and e["attachment"].get("toolUseID") == "toolu_w0001")]
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.HALT, audit.ENVIRONMENTAL, "hookMissing")
    assert found(result) == [(audit.ENVIRONMENTAL, "hookMissing")]


def test_a_pre_hook_input_validation_error_is_counted_not_hook_missing(tmp_path: Path) -> None:
    """No attachment, `is_error` and no `permissionDecision` (attempt 1's validation error):
    the call was refused before any hook ran, so it did nothing (P3-m2)."""
    run = clean_run()
    insert_before_reply(run, tool_use("toolu_v0001", "Write", {"file_path": ANS}),
                        recorded("toolu_v0001"))
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.COUNT, audit.RUN_ATTRIBUTABLE, "inputInvalid")
    assert found(result) == [(audit.RUN_ATTRIBUTABLE, "inputInvalid")]


def test_a_validation_error_shape_with_a_guard_entry_is_hook_missing(tmp_path: Path) -> None:
    """The counted row needs no attachment: with the guard's entry the shape is unexplained."""
    run = clean_run()
    insert_before_reply(run, tool_use("toolu_v0001", "Write", {"file_path": ANS}),
                        guard_entry("toolu_v0001", "Write"), recorded("toolu_v0001"))
    assert found(audit_of(tmp_path, run)) == [(audit.ENVIRONMENTAL, "hookMissing")]


def test_only_a_hook_entry_with_the_declared_guard_command_is_the_guard(tmp_path: Path) -> None:
    """P8-F4: another hook's JSON `allow` (a user-level hook while the guard is off) is not the
    guard's decision."""
    run = clean_run()
    assert run.agent is not None
    for entry in run.agent:
        if entry["type"] == "attachment" and entry["attachment"]["type"] == "hook_success":
            entry["attachment"]["command"] = "python3 /home/user/.claude/other_hook.py"
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.HALT, audit.ENVIRONMENTAL, "hookMissing")


@pytest.mark.parametrize("hook", [
    {"hookEvent": "PostToolUse"},
    {"stdout": ""},
    {"stdout": "not json"},
    {"stdout": json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse"}})},
    {"stdout": json.dumps({"hookSpecificOutput": {"permissionDecision": "deny"}})},
])
def test_a_guard_command_entry_without_an_allow_decision_is_not_a_guard_entry(
    tmp_path: Path, hook: dict[str, Any]
) -> None:
    run = clean_run()
    assert run.agent is not None
    run.agent[5]["attachment"].update(hook)
    assert found(audit_of(tmp_path, run)) == [(audit.ENVIRONMENTAL, "hookMissing")]


def test_two_allowing_guard_entries_are_clean(tmp_path: Path) -> None:
    """A duplicate wiring must not void a run (plan6 §3.4 "Multiple guard entries")."""
    run = clean_run()
    assert run.agent is not None
    run.agent.insert(6, copy.deepcopy(run.agent[5]))
    assert chosen(audit_of(tmp_path, run)) == (audit.CLEAN, None, None)


# == <synthetic> =============================================================================

def test_a_rate_limit_synthetic_is_a_usage_stop(tmp_path: Path) -> None:
    run = clean_run()
    run.append(synthetic("rate_limit"))
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.WAIT, audit.USAGE_STOP, "rateLimit")
    assert found(result) == [(audit.USAGE_STOP, "rateLimit")]


def test_a_rate_limit_synthetic_without_a_tool_result_is_a_usage_stop_not_incomplete(
    tmp_path: Path,
) -> None:
    run = clean_run()
    drop_tool_result(run)
    run.append(synthetic("rate_limit"))
    assert found(audit_of(tmp_path, run)) == [(audit.USAGE_STOP, "rateLimit")]


def test_usage_stops_are_uncapped(tmp_path: Path) -> None:
    run = clean_run()
    run.append(synthetic("rate_limit"))
    assert audit_of(tmp_path, run, earlier=10).action == audit.WAIT


def test_a_rate_limit_with_an_unknown_status_is_a_usage_stop(tmp_path: Path) -> None:
    """M1: a subagent stopped at the limit may hand back a shape never seen."""
    run = clean_run()
    run.append(synthetic("rate_limit"))
    spawn_result(run)["toolUseResult"]["status"] = "paused"
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.WAIT, audit.USAGE_STOP, "rateLimit")
    assert found(result) == [(audit.USAGE_STOP, "rateLimit"),
                             (audit.ENVIRONMENTAL, "spawnShapeUnknown")]


def test_a_rate_limit_with_async_launched_halts(tmp_path: Path) -> None:
    run = clean_run()
    background(run, notified=False)
    run.append(synthetic("rate_limit"))
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.HALT, audit.ENVIRONMENTAL, "backgroundSpawn")


def test_an_authentication_failed_synthetic_halts(tmp_path: Path) -> None:
    run = clean_run()
    run.append(synthetic("authentication_failed"))
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.HALT, audit.ENVIRONMENTAL, "syntheticError")
    assert found(result) == [(audit.ENVIRONMENTAL, "syntheticError")]


def test_a_non_error_synthetic_is_ignored(tmp_path: Path) -> None:
    run = clean_run()
    run.append(synthetic(None))
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.CLEAN, None, None)


def test_a_non_error_synthetic_does_not_hide_an_incomplete_run(tmp_path: Path) -> None:
    run = clean_run()
    drop_tool_result(run)
    run.append(synthetic(None))
    assert found(audit_of(tmp_path, run)) == [(audit.INCOMPLETE, "incomplete")]


# == Run-attributable rows ===================================================================

def test_a_tool_result_with_the_max_turns_shape_is_counted(tmp_path: Path) -> None:
    """Provisional shape until U5.7's smoke records the real one (plan4 §3.3, §4.1 U5.7): the
    driver holds a `tool_result`, and the subagent's last conversation entry is not a final
    `end_turn` reply."""
    run = clean_run()
    assert run.agent is not None
    run.agent = run.agent[:-2]
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.COUNT, audit.RUN_ATTRIBUTABLE, "maxTurns")
    assert found(result) == [(audit.RUN_ATTRIBUTABLE, "maxTurns")]


# == The guard bound (plan6 §3.3, §3.4) =====================================================

OUTSIDE_H6 = "/campaign/exchange-outside/h6-outside.txt"


@pytest.mark.parametrize("name, tool_input", [
    ("Bash", {"command": "ls"}),
    ("Edit", {"file_path": ANS, "old_string": "a", "new_string": "b"}),
    ("Read", {"file_path": "/campaign/store/ledger.jsonl"}),
    ("Read", {"file_path": "/campaign/exchange/ans/q7m2x4k9p3r8t6v5.1.jsonl"}),
    ("Write", {"file_path": REQ, "content": "x"}),
    ("Write", {"file_path": "/campaign/exchange/ans/../req/q7m2x4k9p3r8t6v5.jsonl"}),
    ("Write", {"file_path": OUTSIDE_H6, "content": "ok"}),
])
def test_a_call_the_guard_allowed_outside_its_bounds_halts_as_guard_breach(
    tmp_path: Path, name: str, tool_input: dict[str, Any]
) -> None:
    """H6: a guard `allow` is the permission decision, so the guard is the only bound."""
    run = clean_run()
    insert_before_reply(run, *tool_call("toolu_p0001", name, tool_input))
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.HALT, audit.ENVIRONMENTAL, "guardBreach")
    assert found(result) == [(audit.ENVIRONMENTAL, "guardBreach")]


def test_a_driver_tool_other_than_agent_allowed_by_the_guard_halts(tmp_path: Path) -> None:
    run = clean_run()
    extra = [{**entry, "isSidechain": False} for entry in tool_call(
        "toolu_b0001", "Bash", {"command": "ls"})]
    run.driver[4:4] = extra
    assert found(audit_of(tmp_path, run)) == [(audit.ENVIRONMENTAL, "guardBreach")]


def test_a_rejected_or_refused_call_outside_the_bounds_is_no_breach(tmp_path: Path) -> None:
    """Neither call ran: the reject halts as itself, the refused input is counted."""
    run = clean_run()
    insert_before_reply(run, tool_use("toolu_g0001", "Write", {"file_path": OUTSIDE_H6}),
                        recorded("toolu_g0001"))
    assert found(audit_of(tmp_path / "reject", run)) == [(audit.ENVIRONMENTAL, "guardRejected")]
    run = clean_run()
    insert_before_reply(run, tool_use("toolu_v0001", "Write", {"file_path": OUTSIDE_H6}),
                        recorded("toolu_v0001"))
    assert found(audit_of(tmp_path / "invalid", run)) == [(audit.RUN_ATTRIBUTABLE, "inputInvalid")]


@pytest.mark.parametrize("name, path", [
    ("Read", "/campaign/exchange/req/other00000000000.jsonl"),
    ("Write", "/campaign/exchange/ans/other00000000000.jsonl"),
    ("Write", "/campaign/exchange/ans/q7m2x4k9p3r8t6v5.02.jsonl"),
    ("Write", "/campaign/exchange/ans/q7m2x4k9p3r8t6v5.0.jsonl"),
])
def test_an_allowed_path_inside_the_bounds_but_not_its_own_is_counted(
    tmp_path: Path, name: str, path: str
) -> None:
    run = clean_run()
    insert_before_reply(run, *tool_call("toolu_p0001", name, {"file_path": path}))
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.COUNT, audit.RUN_ATTRIBUTABLE, "pathOutOfBounds")
    assert found(result) == [(audit.RUN_ATTRIBUTABLE, "pathOutOfBounds")]


@pytest.mark.parametrize("name, path", [
    ("Write", "/campaign/exchange/ans/q7m2x4k9p3r8t6v5.1.jsonl"),
    ("Write", "/campaign/exchange/ans/q7m2x4k9p3r8t6v5.12.jsonl"),
    ("Write", "ans/q7m2x4k9p3r8t6v5.jsonl"),
    ("Read", REQ),
    ("Read", "req/../req/q7m2x4k9p3r8t6v5.jsonl"),
])
def test_its_own_request_and_answer_parts_are_in_bounds(
    tmp_path: Path, name: str, path: str
) -> None:
    run = clean_run()
    insert_before_reply(run, *tool_call("toolu_p0001", name, {"file_path": path}))
    assert chosen(audit_of(tmp_path, run)) == (audit.CLEAN, None, None)


# -- the bounds axes (plan6 §5 U5.7, shared with U5.5), on a real tree ------------------------

@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """`exchange/{req,ans,ans-x}`, an `outside/` folder, and the symlinks the axes plant."""
    exchange = tmp_path / "exchange"
    for folder in ("req", "ans", "ans-x"):
        (exchange / folder).mkdir(parents=True)
    (tmp_path / "outside").mkdir()
    (exchange / "ans" / "out").symlink_to(tmp_path / "outside", target_is_directory=True)
    (exchange / "ans" / "leak.jsonl").symlink_to(tmp_path / "outside" / "leak.jsonl")
    (exchange / "alias").symlink_to(exchange / "ans", target_is_directory=True)
    return tmp_path


def in_tree(run: Run, root: Path) -> tuple[Run, audit.Expected]:
    """The clean run with `/campaign` moved under `root`."""
    def move(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [json.loads(json.dumps(e).replace("/campaign", str(root))) for e in entries]
    assert run.agent is not None
    moved = Run(driver=move(run.driver), agent=move(run.agent), meta=run.meta)
    return moved, replace(EXPECTED, request_path=REQ.replace("/campaign", str(root)),
                          answer_path=ANS.replace("/campaign", str(root)))


#: (tool, `file_path` with `<x>` for the exchange folder, expected reason or None for clean).
AXES = [
    ("Write", "<x>/ans/q7m2x4k9p3r8t6v5.jsonl", None),             # absolute
    ("Write", "ans/q7m2x4k9p3r8t6v5.jsonl", None),                 # relative to cwd
    ("Write", "../exchange/ans/q7m2x4k9p3r8t6v5.jsonl", None),     # relative with `..`, inside
    ("Write", "<x>/ans/../req/q7m2x4k9p3r8t6v5.jsonl", "guardBreach"),   # `..` out of ans
    ("Write", "ans/../../outside/f.jsonl", "guardBreach"),          # relative `..` out of exchange
    ("Write", "<x>/ans/leak.jsonl", "guardBreach"),                 # symlinked file, outside
    ("Write", "<x>/ans/out/q7m2x4k9p3r8t6v5.jsonl", "guardBreach"),  # symlinked dir, outside
    ("Write", "<x>/alias/q7m2x4k9p3r8t6v5.jsonl", None),            # symlinked dir, into ans
    ("Write", "<x>/ans-x/q7m2x4k9p3r8t6v5.jsonl", "guardBreach"),    # sibling prefix
    ("Write", "<x>/ans", "guardBreach"),                            # the bound directory itself
    ("Write", "<x>/ans/", "guardBreach"),
    ("Write", "ans", "guardBreach"),
    ("Write", "", "guardBreach"),                                   # empty
    ("Write", None, "guardBreach"),                                 # non-string
    ("Write", 7, "guardBreach"),
    ("Read", "<x>/req/q7m2x4k9p3r8t6v5.jsonl", None),
    ("Read", "req/q7m2x4k9p3r8t6v5.jsonl", None),
    ("Read", "<x>/req", "guardBreach"),
    ("Read", "<x>/req-x/q7m2x4k9p3r8t6v5.jsonl", "guardBreach"),
    ("Read", "<x>/req/../ans/q7m2x4k9p3r8t6v5.jsonl", "guardBreach"),
    ("Read", "<x>/alias/q7m2x4k9p3r8t6v5.jsonl", "guardBreach"),
]


@pytest.mark.parametrize("name, path, reason", AXES)
def test_the_bounds_axes_against_the_audits_own_bounds_code(
    tree: Path, name: str, path: Any, reason: str | None
) -> None:
    run, expected = in_tree(clean_run(), tree)
    exchange = str(tree / "exchange")
    file_path = path.replace("<x>", exchange) if isinstance(path, str) else path
    insert_before_reply(run, *[json.loads(json.dumps(e).replace("/campaign", str(tree)))
                               for e in tool_call("toolu_p0001", name, {"file_path": file_path})])
    result = audit.audit_run(run.stage(tree / "t"), expected, earlier_no_attempt=0,
                             launch=LAUNCH)
    assert result.reason == reason
    assert found(result) == ([] if reason is None else [(audit.ENVIRONMENTAL, reason)])


# == Terminal shape (plan6 §3.4) =============================================================

HANDBACK = "SubagentHandback"


def hand_back(run: Run, *, guard: bool = True, tool_use_id: str = "toolu_h0001") -> None:
    """Replace the final `done` reply with one `SubagentHandback` call and its result."""
    assert run.agent is not None
    run.agent = run.agent[:-1]
    run.append(*tool_call(tool_use_id, HANDBACK, {"result": "done"}, guard=guard,
                          message_id=f"msg_{tool_use_id}"))


@pytest.mark.parametrize("guard", [True, False])
def test_a_run_ending_in_one_hand_back_is_clean(tmp_path: Path, guard: bool) -> None:
    """P8-F2: hooked-allow or unhooked; exempt from the bounds and hook checks."""
    run = clean_run()
    hand_back(run, guard=guard)
    assert chosen(audit_of(tmp_path, run)) == (audit.CLEAN, None, None)


def test_an_unhooked_hand_back_with_no_permission_decision_is_clean(tmp_path: Path) -> None:
    run = clean_run()
    hand_back(run, guard=False)
    assert run.agent is not None
    del run.agent[-1]["permissionDecision"]
    assert chosen(audit_of(tmp_path, run)) == (audit.CLEAN, None, None)


def test_two_hand_backs_are_counted(tmp_path: Path) -> None:
    run = clean_run()
    hand_back(run)
    run.append(*tool_call("toolu_h0002", HANDBACK, {"result": "done"}, message_id="msg_h2"))
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.COUNT, audit.RUN_ATTRIBUTABLE, "toolOutOfBounds")
    assert found(result) == [(audit.RUN_ATTRIBUTABLE, "toolOutOfBounds")]


def test_a_write_after_the_hand_back_is_counted(tmp_path: Path) -> None:
    run = clean_run()
    hand_back(run)
    run.append(*tool_call("toolu_w0002", "Write", {"file_path": ANS, "content": "x"},
                          message_id="msg_w2"))
    result = audit_of(tmp_path, run)
    assert result.action == audit.COUNT
    assert (audit.RUN_ATTRIBUTABLE, "toolOutOfBounds") in found(result)


def test_a_hand_back_refused_by_input_validation_is_counted(tmp_path: Path) -> None:
    """P9-F3: the hand-back is exempt only from the bounds checks and `hookMissing`, so the
    validation row counts it (attempt 1's recorded validation error, re-keyed)."""
    run = clean_run()
    assert run.agent is not None
    run.agent = run.agent[:-1]
    run.append(tool_use("toolu_v0001", HANDBACK, {"result": "done"}), recorded("toolu_v0001"))
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.COUNT, audit.RUN_ATTRIBUTABLE, "inputInvalid")
    assert found(result) == [(audit.RUN_ATTRIBUTABLE, "inputInvalid")]


def test_a_rejected_hand_back_halts_as_guard_rejected(tmp_path: Path) -> None:
    run = clean_run()
    assert run.agent is not None
    run.agent = run.agent[:-1]
    denied = copy.deepcopy(recorded("toolu_g0001"))
    denied["message"]["content"][0]["tool_use_id"] = "toolu_h0001"
    run.append(tool_use("toolu_h0001", HANDBACK, {"result": "done"}), denied)
    assert chosen(audit_of(tmp_path, run)) == (audit.HALT, audit.ENVIRONMENTAL, "guardRejected")


# == Precedence ==============================================================================

def guard_denied(run: Run) -> None:
    """U6.7's D1 (a guard deny) planted before the reply: a `guardRejected` row."""
    insert_before_reply(run, tool_use("toolu_g0001", "Write", {"file_path": ANS, "content": "ok"}),
                        recorded("toolu_g0001"))


def test_a_refusal_with_a_guard_deny_halts_as_a_safeguards_stop(tmp_path: Path) -> None:
    """P9-F2: plan6's halt rows decide the reason in table order, and the reason decides what the
    stakeholder may do at `resolve` (`--skip-batch` is for a safeguards flag, plan6 §3.5)."""
    run = clean_run()
    guard_denied(run)
    insert_before_reply(run, refusal())
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.HALT, audit.ENVIRONMENTAL, "safeguardStop")
    assert found(result) == [(audit.ENVIRONMENTAL, "safeguardStop"),
                             (audit.ENVIRONMENTAL, "guardRejected")]


def test_a_background_spawn_with_a_guard_deny_halts_as_a_background_spawn(tmp_path: Path) -> None:
    run = clean_run()
    background(run, notified=False)
    guard_denied(run)
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.HALT, audit.ENVIRONMENTAL, "backgroundSpawn")
    assert found(result) == [(audit.ENVIRONMENTAL, "backgroundSpawn"),
                             (audit.ENVIRONMENTAL, "guardRejected")]


def test_a_halt_row_beats_a_counted_row_that_comes_first_in_the_table(tmp_path: Path) -> None:
    """P8-F3: the counted row precedes the halt row in table order, so only the tiers halt it."""
    run = clean_run()
    insert_before_reply(run, *invalid_call())
    run.append(synthetic("authentication_failed"))
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.HALT, audit.ENVIRONMENTAL, "syntheticError")
    assert found(result) == [(audit.RUN_ATTRIBUTABLE, "inputInvalid"),
                             (audit.ENVIRONMENTAL, "syntheticError")]


def test_a_usage_stop_beats_a_counted_row_that_comes_first_in_the_table(tmp_path: Path) -> None:
    """P8-F3's A6 pair, re-planted on a counted row now that a reject halts."""
    run = clean_run()
    insert_before_reply(run, *invalid_call())
    run.append(synthetic("rate_limit"))
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.WAIT, audit.USAGE_STOP, "rateLimit")
    assert found(result) == [(audit.RUN_ATTRIBUTABLE, "inputInvalid"),
                             (audit.USAGE_STOP, "rateLimit")]


def test_a_reissue_row_beats_a_counted_row(tmp_path: Path) -> None:
    run = clean_run()
    drop_tool_result(run)
    insert_before_reply(run, *invalid_call())
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.REISSUE, audit.INCOMPLETE, "incomplete")
    assert found(result) == [(audit.RUN_ATTRIBUTABLE, "inputInvalid"),
                             (audit.INCOMPLETE, "incomplete")]


def test_a_halt_row_beats_a_reissue_row(tmp_path: Path) -> None:
    run = clean_run()
    drop_tool_result(run)
    run.assistant("msg_s0004")[0]["message"]["model"] = "claude-sonnet-5-0"
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.HALT, audit.ENVIRONMENTAL, "modelMismatch")
    assert found(result) == [(audit.INCOMPLETE, "incomplete"),
                             (audit.ENVIRONMENTAL, "modelMismatch")]


def test_within_a_tier_the_first_row_in_table_order_is_chosen(tmp_path: Path) -> None:
    run = clean_run()
    run.spawn_input()["isolation"] = "worktree"
    run.assistant("msg_s0004")[0]["message"]["model"] = "claude-sonnet-5-0"
    insert_before_reply(run, *invalid_call(), *tool_call("toolu_p0001", "Read", {
        "file_path": "/campaign/exchange/req/other00000000000.jsonl"}))
    result = audit_of(tmp_path, run)
    assert chosen(result) == (audit.HALT, audit.DRIVER, "toolUseOverride")
    assert found(result) == [(audit.RUN_ATTRIBUTABLE, "inputInvalid"),
                             (audit.DRIVER, "toolUseOverride"),
                             (audit.ENVIRONMENTAL, "modelMismatch"),
                             (audit.RUN_ATTRIBUTABLE, "pathOutOfBounds")]


@pytest.mark.parametrize("which", ["driver", "agent"])
def test_a_truncated_final_line_of_a_killed_run_is_read_as_truncation(
    tmp_path: Path, which: str
) -> None:
    """P8-F5: a kill mid-write leaves a final line with no newline; the run is incomplete."""
    run = clean_run()
    drop_tool_result(run)
    driver = run.stage(tmp_path)
    path = driver if which == "driver" else (
        driver.with_suffix("") / "subagents" / f"agent-{AGENT}.jsonl")
    path.write_text(path.read_text("utf-8") + '{"parentUuid":"a00', "utf-8")
    result = audit.audit_run(driver, EXPECTED, earlier_no_attempt=0, launch=LAUNCH)
    assert chosen(result) == (audit.REISSUE, audit.INCOMPLETE, "incomplete")


def test_an_unreadable_transcript_line_halts(tmp_path: Path) -> None:
    run = clean_run()
    driver = run.stage(tmp_path)
    agent_file = driver.with_suffix("") / "subagents" / f"agent-{AGENT}.jsonl"
    agent_file.write_text(agent_file.read_text("utf-8") + "{not json\n", "utf-8")
    result = audit.audit_run(driver, EXPECTED, earlier_no_attempt=0, launch=LAUNCH)
    assert chosen(result) == (audit.HALT, audit.ENVIRONMENTAL, "transcriptUnreadable")


# == Tokens and the runs[] record ============================================================

def test_one_message_id_over_three_entries_counts_its_last_entrys_usage() -> None:
    """Hand-written: `msg_s0003` spans three entries with `output_tokens` 3, 900, 2087 and
    `cache_creation_input_tokens` 500, 800, 650 (so last is neither first, max nor sum)."""
    entries = [e for e in read_jsonl(FIXTURES / f"agent-{AGENT}.jsonl")
               if e.get("type") == "assistant" and e["message"]["id"] == "msg_s0003"]
    assert [e["message"]["usage"]["output_tokens"] for e in entries] == [3, 900, 2087]
    assert audit.count_tokens(entries) == {"input": 2, "cacheWrite": 650, "cacheRead": 1500,
                                           "output": 2087}


def test_run_tokens_sum_the_last_usage_of_every_message_id(tmp_path: Path) -> None:
    # Hand-computed over the four message ids of the clean run:
    # input 3+2+2+2, cacheWrite 1200+300+650+100, cacheRead 0+1200+1500+2300, output 60+45+2087+4.
    result = audit_of(tmp_path, clean_run())
    assert result.facts is not None
    assert result.facts.tokens == {"input": 9, "cacheWrite": 2250, "cacheRead": 5000,
                                   "output": 2196}


def test_synthetic_entries_add_no_tokens(tmp_path: Path) -> None:
    run = clean_run()
    run.append(synthetic(None))
    facts = audit_of(tmp_path, run).facts
    assert facts is not None and facts.tokens["output"] == 2196


def test_the_run_facts_come_from_the_located_transcript(tmp_path: Path) -> None:
    run = clean_run()
    insert_before_reply(run, attachment("instructions", files=[
        {"path": "/campaign/CLAUDE.md", "type": "Project", "content": "Be brief."}]))
    facts = audit_of(tmp_path, run).facts
    assert facts is not None
    context = {"userEmail": "labeler@example.invalid"}
    canonical = json.dumps(context, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert facts == audit.RunFacts(
        agent_id=AGENT,
        claude_code_version="2.1.316",
        model="claude-sonnet-5-5",
        instructions=(("/campaign/CLAUDE.md", hashlib.sha256(b"Be brief.").hexdigest()),),
        session_context_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        thinking={"type": "adaptive"},
        effort="high",
        tokens={"input": 9, "cacheWrite": 2250, "cacheRead": 5000, "output": 2196},
    )


RUNS_KEYS = {"runId", "role", "batchId", "attempt", "status", "voidClass", "voidReason",
             "claudeCodeVersion", "agentId", "model", "requestSha256", "instructions",
             "sessionContextSha256", "thinking", "effort", "tokens"}


def record_for(result: audit.Audit, outcome: Any = None) -> dict[str, Any] | None:
    return audit.run_record(result, audit.combine(result, outcome), run_id="run_01",
                            role="label", batch_id="b_01", attempt=1, request_sha256="f" * 64)


def test_the_runs_record_holds_exactly_plan4_s_keys(tmp_path: Path) -> None:
    result = audit_of(tmp_path, clean_run())
    record = record_for(result, ingest.Ingested(answers={}, answer_sha256="a" * 64))
    assert record is not None and set(record) == RUNS_KEYS
    assert record["status"] == "ingested" and record["voidClass"] is None
    assert record["instructions"] == [] and record["tokens"]["output"] == 2196
    assert (record["runId"], record["role"], record["batchId"], record["attempt"],
            record["requestSha256"]) == ("run_01", "label", "b_01", 1, "f" * 64)
    assert record["agentId"] == AGENT and record["claudeCodeVersion"] == "2.1.316"


def test_a_halted_run_is_recorded_as_a_void_with_its_class_and_reason(tmp_path: Path) -> None:
    run = clean_run()
    run.append(synthetic("authentication_failed"))
    record = record_for(audit_of(tmp_path, run))
    assert record is not None
    assert (record["status"], record["voidClass"], record["voidReason"]) == (
        "void", audit.ENVIRONMENTAL, "syntheticError")


@pytest.mark.parametrize("edit, status", [
    (lambda run: run.append(synthetic("rate_limit")), "usage-stop"),
    (drop_tool_result, "incomplete"),
])
def test_no_attempt_runs_are_recorded_with_their_status(tmp_path: Path, edit, status) -> None:
    run = clean_run()
    edit(run)
    record = record_for(audit_of(tmp_path, run))
    assert record is not None and record["status"] == status


def test_a_run_never_spawned_has_no_runs_record(tmp_path: Path) -> None:
    run = clean_run()
    run.driver = [run.driver[0]]
    assert record_for(audit_of(tmp_path, run)) is None


# == The U5.6 seam: combining the audit with ingest_answer ===================================

def test_only_clean_and_counted_audits_ingest_the_answer(tmp_path: Path) -> None:
    cases = {"clean": clean_run(), "count": clean_run(), "halt": clean_run(),
             "reissue": clean_run(), "wait": clean_run()}
    insert_before_reply(cases["count"], *invalid_call())
    cases["halt"].append(synthetic("authentication_failed"))
    drop_tool_result(cases["reissue"])
    cases["wait"].append(synthetic("rate_limit"))
    ingests = {name: audit_of(tmp_path / name, run).ingests_answer for name, run in cases.items()}
    assert ingests == {"clean": True, "count": True, "halt": False, "reissue": False,
                       "wait": False}


def test_a_clean_audit_takes_ingests_outcome(tmp_path: Path) -> None:
    result = audit_of(tmp_path, clean_run())
    done = audit.combine(result, ingest.Ingested(answers={}, answer_sha256="a" * 64))
    assert (done.status, done.void_class, done.void_reason, done.answer_sha256) == (
        "ingested", None, None, "a" * 64)
    assert not done.halts and not done.counts_attempt
    void = audit.combine(result, ingest.Void(reason="keyMissing", details=("k",),
                                             answer_sha256="b" * 64))
    assert (void.status, void.void_class, void.void_reason, void.answer_sha256) == (
        "void", audit.RUN_ATTRIBUTABLE, "keyMissing", "b" * 64)
    assert void.counts_attempt and not void.halts


def test_a_counted_audit_beats_a_valid_answer_and_keeps_its_hash(tmp_path: Path) -> None:
    run = clean_run()
    insert_before_reply(run, *invalid_call())
    result = audit_of(tmp_path, run)
    done = audit.combine(result, ingest.Ingested(answers={}, answer_sha256="a" * 64))
    assert (done.status, done.void_class, done.void_reason, done.answer_sha256) == (
        "void", audit.RUN_ATTRIBUTABLE, "inputInvalid", "a" * 64)
    assert done.counts_attempt


def test_audits_that_do_not_ingest_record_no_answer_hash(tmp_path: Path) -> None:
    run = clean_run()
    drop_tool_result(run)
    done = audit.combine(audit_of(tmp_path, run), None)
    assert (done.status, done.answer_sha256, done.counts_attempt, done.halts) == (
        "incomplete", None, False, False)
    assert done.reissues_under_new_name


def test_combine_refuses_an_outcome_the_audit_does_not_take(tmp_path: Path) -> None:
    halted = clean_run()
    halted.append(synthetic("authentication_failed"))
    halt = audit_of(tmp_path / "h", halted)
    with pytest.raises(ValueError):
        audit.combine(halt, ingest.Ingested(answers={}, answer_sha256="a" * 64))
    clean = audit_of(tmp_path / "c", clean_run())
    with pytest.raises(ValueError):
        audit.combine(clean, None)
    with pytest.raises(ValueError):
        audit.combine(clean, ingest.AlreadyIngested(answer_sha256="a" * 64))
