"""The transcript audit (plan `docs/plans/typed-decision-models6.md` §3.3 guard bounds and §3.4
rows, over plan4 `docs/plans/typed-decision-models4.md` §3.3 "Locating the run", its other rows
and caps, and §3.7 `runs[]` and the token rule; method note
`docs/plans/typed-decision-models-ml.md` §Q1.2 "Ingest audit", "Reruns").

**Locating the run** never uses what it checks. `audit_run` reads the driver session's transcript
(`<project>/<session>.jsonl`, the runner's `--session-id`) for the `Agent` `tool_use` whose
`input.prompt` contains the request path, then the `<project>/<session>/subagents/
agent-<id>.meta.json` whose `toolUseId` is that `tool_use`'s id, and reads the `agent-<id>.jsonl`
beside it. Only conversation entries count; an `api-request-blob` replay is not a second spawn.
No transcript file, or a child that exited non-zero leaving neither an `Agent` call nor a
`<synthetic>`, is `launchFailed` (`Launch` carries the exit code and the ledger's clock).

**Completion** is F2's foreground shape only: the driver's `tool_result` matched by the spawn's
`tool_use_id` has `toolUseResult.status: "completed"`, and the meta has `requestShape:
"foreground"` and `finished: true`. `async_launched` or a `background` meta is `backgroundSpawn`;
any other shape with a result is `spawnShapeUnknown`. With no matched result and no error
`<synthetic>` in either transcript the run is `incomplete`, and every tool use with no
`tool_result` is **in flight**: no other row reads it, so a killed run re-issues.

**Guard decisions** (plan6 §3.4), per tool use of either transcript, keyed on its `tool_result`
entry's top-level `permissionDecision`:
- `reject`/`hook` → `guardRejected`, a guard deny or a blocked guard failure (one class: report
  §10 shows no structure separates them);
- `accept`/`hook` with a **guard entry** (a `PreToolUse` `hook_success` for its id from
  `Expected.guard_command`, whose `stdout` is the guard's `allow`) → allowed;
- none, `is_error` and no attachment for its id → `inputInvalid`, counted;
- anything else → `hookMissing`. `SubagentHandback` is exempt from this row.
An allowed call is re-checked against the guard's bounds with this module's own canonicalization
(`os.path.realpath`, relative paths joined to the entry's `cwd`; strictly inside the request's
folder for `Read`, the answer's for `Write`; `Agent` only for the driver) → else `guardBreach`.
Inside them but not this run's request or answer parts (`<opaque>.jsonl`, `<opaque>.<k>.jsonl`,
`k` without a leading zero) is `pathOutOfBounds`, counted.

**Findings.** Every matching row is a `Finding(void_class, reason, detail)`, in `ROWS` order
(plan6's table, then plan4's). The classes: `DRIVER`, `ENVIRONMENTAL` halt (`HALT`);
`NEVER_SPAWNED`, `INCOMPLETE`, `LAUNCH_FAILED` re-issue with no attempt (`REISSUE`); `USAGE_STOP`
waits for the next window (`WAIT`); `RUN_ATTRIBUTABLE` is counted (`COUNT`). The caller passes
`earlier_no_attempt`, the batch's earlier never-spawned plus incomplete outcomes; from the second
on such a finding is `DRIVER` with the same reason, except an `incomplete` the caller marks
`interrupted`. `launchFailed` is uncapped until launches have produced nothing for longer than
`Launch.fail_bound`, then `launchFailing` halts.
**Precedence** (plan6 §3.4): halts, then the usage stop, then `spawnShapeUnknown`, then the
re-issues, then counted rows; within a tier the first row in `ROWS` order. No finding is `CLEAN`.
A `<synthetic>` entry with no `error` is ignored throughout.

Choices this module makes where the plans leave room, each failing closed:
- two `tool_use`s naming the request (`toolUseRepeated`) and an unparseable line other than a
  final line with no newline (a write a kill cut short, P8-F5) halt;
- a run with no recorded effort is `effortOverride`;
- **`maxTurns` is provisional** until U5.7's smoke records its shape: a completed run whose last
  assistant entry is neither an `end_turn` reply nor exactly one `SubagentHandback` followed only
  by its `tool_result`. More than one hand-back, or one that is not the last tool use, is
  `toolOutOfBounds`, counted.

**The U5.6 seam.** `ingest.ingest_answer` returns only answer-check outcomes; this module's
classes take precedence, and `combine(audit, outcome)` joins them. `Audit.ingests_answer` is true
only for `CLEAN` and `COUNT`: the caller then runs `ingest_answer` and passes its `Ingested` or
`Void`, and the `Verdict` carries that outcome's `answer_sha256`. For `HALT`, `REISSUE` and
`WAIT` the caller passes `None` and the verdict records **no** `answerSha256`; re-issued and
usage-stop batches re-issue under a new opaque name (`reissues_under_new_name`). `run_record`
builds §3.7's `runs[]` entry; a run with no located transcript has none.

**Tokens** (§3.7): per `message.id`, the `usage` of the **last** entry carrying it (plan4 §2: one
message spans several entries whose `usage` can differ), summed over the run's non-`<synthetic>`
messages as `{input, cacheWrite, cacheRead, output}`.
"""


from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .ingest import AlreadyIngested, Ingested, Void

HALT, REISSUE, WAIT, COUNT, CLEAN = "halt", "reissue", "wait", "count", "clean"

DRIVER = "driver"
ENVIRONMENTAL = "environmental"
NEVER_SPAWNED = "never-spawned"
LAUNCH_FAILED = "launch-failed"
INCOMPLETE = "incomplete"
USAGE_STOP = "usage-stop"
RUN_ATTRIBUTABLE = "run-attributable"

#: The rows in table order (plan6 §3.4's, then plan4 §3.3's): reason → its class. A capped
#: never-spawned or incomplete finding keeps its reason and takes `DRIVER`.
ROWS: Mapping[str, str] = {
    "safeguardStop": ENVIRONMENTAL,
    "backgroundSpawn": ENVIRONMENTAL,
    "guardRejected": ENVIRONMENTAL,
    "guardBreach": ENVIRONMENTAL,
    "inputInvalid": RUN_ATTRIBUTABLE,
    "hookMissing": ENVIRONMENTAL,
    "driverModelMismatch": ENVIRONMENTAL,
    "syntheticError": ENVIRONMENTAL,
    "rateLimit": USAGE_STOP,
    "spawnShapeUnknown": ENVIRONMENTAL,
    "incomplete": INCOMPLETE,
    "neverSpawned": NEVER_SPAWNED,
    "launchFailed": LAUNCH_FAILED,
    "launchFailing": ENVIRONMENTAL,
    "toolUseRepeated": DRIVER,
    "transcriptMissing": DRIVER,
    "transcriptAmbiguous": DRIVER,
    "transcriptUnreadable": ENVIRONMENTAL,
    "spawnPromptMismatch": DRIVER,
    "toolUseOverride": DRIVER,
    "subagentTypeMismatch": DRIVER,
    "agentTypeMismatch": ENVIRONMENTAL,
    "modelMismatch": ENVIRONMENTAL,
    "effortOverride": ENVIRONMENTAL,
    "instructionsMismatch": ENVIRONMENTAL,
    "mcpInstructions": ENVIRONMENTAL,
    "mcpDeferredTool": ENVIRONMENTAL,
    "gitStatus": ENVIRONMENTAL,
    "maxTurns": RUN_ATTRIBUTABLE,
    "toolOutOfBounds": RUN_ATTRIBUTABLE,
    "pathOutOfBounds": RUN_ATTRIBUTABLE,
}

_ACTION = {DRIVER: HALT, ENVIRONMENTAL: HALT, NEVER_SPAWNED: REISSUE, INCOMPLETE: REISSUE,
           LAUNCH_FAILED: REISSUE, USAGE_STOP: WAIT, RUN_ATTRIBUTABLE: COUNT}
_TIER = {HALT: 0, WAIT: 1, REISSUE: 3, COUNT: 4}
#: plan6 §3.4's precedence: halts, the usage stop, then `spawnShapeUnknown`, then the re-issues.
_SHAPE_TIER = 2


def _tier(finding: Finding) -> int:
    if finding.reason == "spawnShapeUnknown":
        return _SHAPE_TIER
    return _TIER[_ACTION[finding.void_class]]


#: `tool_use` input keys only the role's definition may set (plan4 §3.3, §3.4).
OVERRIDE_KEYS = ("model", "effort", "isolation", "run_in_background")
#: The tool a subagent may call once, last, as its reply (P8-F2; plan6 §3.4 "Terminal shape").
HANDBACK = "SubagentHandback"
#: plan6 §3.3: `dm-driver` pins this model.
DRIVER_MODEL = "claude-sonnet-5-5"
SYNTHETIC = "<synthetic>"
RATE_LIMIT = "rate_limit"
_CONVERSATION = ("user", "assistant", "attachment")
_TOKEN_FIELDS = (("input", "input_tokens"), ("cacheWrite", "cache_creation_input_tokens"),
                 ("cacheRead", "cache_read_input_tokens"), ("output", "output_tokens"))


@dataclass(frozen=True)
class Expected:
    """What the run is checked against: the role's pins and declared expectations (U5.7's
    `install`) and this batch's request and answer paths. `spawn_template` holds one
    `{request_path}` slot; `instructions` is the declared `(path, sha256)` set."""

    agent_type: str
    model: str
    effort: str
    spawn_template: str
    request_path: str
    answer_path: str
    guard_command: str
    instructions: frozenset[tuple[str, str]] = frozenset()

    @property
    def spawn_prompt(self) -> str:
        return self.spawn_template.replace("{request_path}", self.request_path)


@dataclass(frozen=True)
class Launch:
    """The launch being audited, from the runner and the ledger (plan6 §3.2, §3.4)."""

    exit_code: int = 0
    at: datetime | None = None
    earlier_failures: tuple[datetime, ...] = ()
    last_produced: datetime | None = None
    fail_bound: timedelta = timedelta(hours=6)


@dataclass(frozen=True)
class Finding:
    void_class: str
    reason: str
    detail: str


@dataclass(frozen=True)
class RunFacts:
    """§3.7's per-run facts read from the located transcript."""

    agent_id: str
    claude_code_version: str | None
    model: str | None
    instructions: tuple[tuple[str, str], ...]
    session_context_sha256: str | None
    thinking: Any
    effort: str | None
    tokens: Mapping[str, int]


@dataclass(frozen=True)
class Audit:
    """The chosen class (`action`, `void_class`, `reason`; all `None` but `action` when `CLEAN`),
    every matching row in `findings`, and the run's facts (`None` with no located transcript)."""

    action: str
    void_class: str | None
    reason: str | None
    findings: tuple[Finding, ...]
    facts: RunFacts | None

    @property
    def ingests_answer(self) -> bool:
        """Whether the caller runs `ingest_answer` and records its `answer_sha256`."""
        return self.action in (CLEAN, COUNT)


@dataclass(frozen=True)
class Verdict:
    """One run's outcome. `status` is §3.7's (`None` for a never-spawned batch, which has no
    run); `answer_sha256` is set only when `ingest_answer` ran."""

    status: str | None
    void_class: str | None
    void_reason: str | None
    details: tuple[str, ...]
    answer_sha256: str | None
    halts: bool
    counts_attempt: bool
    reissues_under_new_name: bool


class _Unreadable(ValueError):
    pass


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text("utf-8")
    lines = text.splitlines()
    entries = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError as exc:
            if number == len(lines) and not text.endswith("\n"):
                break  # a write cut short by a kill (P8-F5)
            raise _Unreadable(f"{path.name} line {number}: {exc}") from exc
        if not isinstance(entry, dict):
            raise _Unreadable(f"{path.name} line {number}: not an object")
        entries.append(entry)
    return entries


def _blocks(entry: Mapping[str, Any], kind: str) -> list[dict[str, Any]]:
    message = entry.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == kind]


def _is_synthetic(entry: Mapping[str, Any]) -> bool:
    message = entry.get("message")
    return isinstance(message, dict) and message.get("model") == SYNTHETIC


def _attachments(entries: Iterable[Mapping[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [e["attachment"] for e in entries if e.get("type") == "attachment"
            and isinstance(e.get("attachment"), dict) and e["attachment"].get("type") == kind]


def _assistant(entries: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [e for e in entries if e.get("type") == "assistant" and not _is_synthetic(e)]


def count_tokens(entries: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """§3.7's token rule over assistant entries (module docstring)."""
    last: dict[str, Mapping[str, Any]] = {}
    for entry in _assistant(entries):
        message = entry["message"]
        usage = message.get("usage")
        last[message.get("id")] = usage if isinstance(usage, dict) else {}
    return {name: sum(int(usage.get(key) or 0) for usage in last.values())
            for name, key in _TOKEN_FIELDS}


def _is_guard_entry(hook: Mapping[str, Any], command: str) -> bool:
    """A `PreToolUse` `hook_success` from the declared guard command whose `stdout` is its `allow`
    JSON (plan6 §3.4 "guard entry"; P8-F4)."""
    if hook.get("hookEvent") != "PreToolUse" or hook.get("command") != command:
        return False
    try:
        out = json.loads(hook.get("stdout") or "")
    except ValueError:
        return False
    specific = out.get("hookSpecificOutput") if isinstance(out, dict) else None
    return isinstance(specific, dict) and specific.get("permissionDecision") == "allow"


def _results(entries: Iterable[Mapping[str, Any]]) -> dict[Any, tuple[Mapping, Mapping]]:
    """Each `tool_use_id` → the `user` entry holding its first `tool_result`, and that block."""
    found: dict[Any, tuple[Mapping, Mapping]] = {}
    for entry in entries:
        if entry.get("type") == "user":
            for block in _blocks(entry, "tool_result"):
                found.setdefault(block.get("tool_use_id"), (entry, block))
    return found


def _decision(entry: Mapping[str, Any]) -> tuple[Any, Any] | None:
    """The guard decision: the `tool_result` entry's top-level `permissionDecision`."""
    decision = entry.get("permissionDecision")
    if not isinstance(decision, dict):
        return None
    return decision.get("decision"), decision.get("source")


def _answer_part_pattern(answer_path: str) -> re.Pattern[str]:
    stem = os.path.realpath(answer_path).removesuffix(".jsonl")
    return re.compile(rf"{re.escape(stem)}(\.[1-9][0-9]*)?\.jsonl")


def _canonical(file_path: Any, cwd: Any) -> str | None:
    """Plan6 §3.3's canonical path: a non-empty string, joined to `cwd` when relative, through
    `os.path.realpath` (symlinks and `..`). `None` when it has none. Recomputed here rather than
    imported from `guard_hook.py`, so one canonicalization defect cannot hide in both (P3-m3)."""
    if not isinstance(file_path, str) or not file_path:
        return None
    if not os.path.isabs(file_path):
        if not isinstance(cwd, str) or not os.path.isabs(cwd):
            return None
        file_path = os.path.join(cwd, file_path)
    return os.path.realpath(file_path)


def _strictly_inside(path: str, folder: str) -> bool:
    return path.startswith(os.path.realpath(folder) + os.sep)


class _Findings:
    def __init__(self, earlier_no_attempt: int, interrupted: bool) -> None:
        self.items: list[Finding] = []
        self.capped = {NEVER_SPAWNED} if interrupted else {NEVER_SPAWNED, INCOMPLETE}
        if earlier_no_attempt < 1:
            self.capped = set()

    def add(self, reason: str, detail: str) -> None:
        void_class = ROWS[reason]
        if void_class in self.capped:
            void_class = DRIVER
            detail += "; a second never-spawned or incomplete outcome for this batch"
        self.items.append(Finding(void_class, reason, detail))

    def audit(self, facts: RunFacts | None) -> Audit:
        ordered = tuple(sorted(self.items, key=lambda f: list(ROWS).index(f.reason)))
        if not ordered:
            return Audit(CLEAN, None, None, (), facts)
        first = min(ordered, key=_tier)
        return Audit(_ACTION[first.void_class], first.void_class, first.reason, ordered, facts)


def audit_run(driver_transcript: Path, expected: Expected, *, earlier_no_attempt: int,
              launch: Launch = Launch(), interrupted: bool = False) -> Audit:
    """Locate and audit the run spawned for `expected.request_path` (module docstring)."""
    found = _Findings(earlier_no_attempt, interrupted)
    if not driver_transcript.is_file():
        _launch_failed(found, launch, f"no transcript {driver_transcript.name}")
        return found.audit(None)
    try:
        driver = _read_jsonl(driver_transcript)
    except _Unreadable as exc:
        found.add("transcriptUnreadable", str(exc))
        return found.audit(None)
    driver_errors = _errors(driver)
    spawns = [
        (entry, block) for entry in driver if entry.get("type") == "assistant"
        for block in _blocks(entry, "tool_use")
        if block.get("name") == "Agent" and isinstance(block.get("input"), dict)
        and isinstance(block["input"].get("prompt"), str)
        and expected.request_path in block["input"]["prompt"]
    ]
    if not spawns:
        _check_session(found, driver, [], driver_errors)
        if driver_errors:
            pass  # a usage stop or a halt, from `_check_session`
        elif launch.exit_code != 0:
            _launch_failed(found, launch, f"exit {launch.exit_code} with no Agent tool_use")
        else:
            found.add("neverSpawned", f"no Agent tool_use names {expected.request_path}")
        return found.audit(None)
    if len({block.get("id") for _, block in spawns}) > 1:
        found.add("toolUseRepeated", f"{len(spawns)} Agent tool_uses name the request")
        return found.audit(None)
    spawn = spawns[0][1]
    spawn_id = spawn.get("id")
    matched = _results(driver).get(spawn_id)
    in_flight = matched is None
    subagents = driver_transcript.with_suffix("") / "subagents"
    metas = []
    for meta_path in sorted(subagents.glob("agent-*.meta.json")) if subagents.is_dir() else []:
        try:
            meta = json.loads(meta_path.read_text("utf-8"))
        except ValueError:
            continue
        if isinstance(meta, dict) and meta.get("toolUseId") == spawn_id:
            metas.append((meta_path, meta))
    if len(metas) > 1:
        found.add("transcriptAmbiguous", f"{len(metas)} subagent transcripts for {spawn_id}")
        return found.audit(None)
    agent_path = metas[0][0].with_name(metas[0][0].name.removesuffix(".meta.json") + ".jsonl") \
        if metas else None
    agent: list[dict[str, Any]] = []
    located = agent_path is not None and agent_path.is_file()
    if located:
        try:
            agent = _read_jsonl(agent_path)
        except _Unreadable as exc:
            found.add("transcriptUnreadable", str(exc))
            return found.audit(None)
    elif not in_flight:
        found.add("transcriptMissing", f"no subagent transcript for {spawn_id}")
    errors = [*driver_errors, *_errors(agent)]

    if in_flight and not errors:
        found.add("incomplete", f"the driver transcript holds no tool_result for {spawn_id}")
    _check_completion(found, matched, metas[0][1] if metas else None)
    driver_allowed = _check_guard(found, expected, driver, _tool_uses(driver), in_flight)
    beyond = sorted({str(call.get("name")) for call in driver_allowed} - {"Agent"})
    if beyond:
        found.add("guardBreach", f"the guard allowed the driver {beyond}")
    _check_session(found, driver, agent, errors)
    if agent_path is None or not located:
        return found.audit(None)
    agent_id = agent_path.name.removeprefix("agent-").removesuffix(".jsonl")
    _check_run(found, expected, spawn, matched, metas[0][1], agent, errors, in_flight)
    return found.audit(_facts(agent_id, agent))


def _tool_uses(entries: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Every `tool_use` block of the non-`<synthetic>` assistant entries, once per id, each with
    its entry's `cwd` under `"_cwd"` (the hook input's `cwd`)."""
    calls: dict[Any, Mapping[str, Any]] = {}
    for entry in _assistant(entries):
        for block in _blocks(entry, "tool_use"):
            calls.setdefault(block.get("id"), {**block, "_cwd": entry.get("cwd")})
    return list(calls.values())


def _check_completion(found: _Findings, matched: tuple[Mapping, Mapping] | None,
                      meta: Mapping[str, Any] | None) -> None:
    """Plan6 §3.4's completion rows: only F2's foreground shape is a completion."""
    outcome = matched[0].get("toolUseResult") if matched is not None else None
    status = outcome.get("status") if isinstance(outcome, dict) else None
    shape = meta.get("requestShape") if meta is not None else None
    if status == "async_launched" or shape == "background":
        found.add("backgroundSpawn", f"status {status!r}, requestShape {shape!r}")
    elif matched is not None and meta is not None and (
            status != "completed" or shape != "foreground" or meta.get("finished") is not True):
        found.add("spawnShapeUnknown", f"status {status!r}, requestShape {shape!r}, "
                                       f"finished {meta.get('finished')!r}")


def _errors(entries: Iterable[Mapping[str, Any]]) -> list[Any]:
    return [e.get("error") for e in entries
            if e.get("type") == "assistant" and _is_synthetic(e) and e.get("error") is not None]


def _launch_failed(found: _Findings, launch: Launch, detail: str) -> None:
    """Plan6 §3.4: uncapped, until launches have produced nothing for longer than the bound,
    counted from the first `launchFailed` after the last launch that produced something."""
    since = [at for at in launch.earlier_failures
             if launch.last_produced is None or at > launch.last_produced]
    if since:
        if launch.at is None:
            raise ValueError("a launch after earlier failures needs its time")
        if launch.at - min(since) > launch.fail_bound:
            found.add("launchFailing", f"{detail}; nothing produced since {min(since).isoformat()}")
            return
    found.add("launchFailed", detail)


def _check_session(found: _Findings, driver: Sequence[Mapping[str, Any]],
                   agent: Sequence[Mapping[str, Any]], errors: Sequence[Any]) -> None:
    """The rows read from both transcripts: safeguards, the driver's pin, `<synthetic>` errors."""
    entries = [*driver, *agent]
    if any(e.get("type") == "assistant" and isinstance(e.get("message"), dict)
           and e["message"].get("stop_reason") == "refusal" for e in entries) or any(
            e.get("type") == "user" and "refusedMessageId" in e for e in entries):
        found.add("safeguardStop", "a refusal stop or a refusedMessageId entry")
    models = sorted({str(e["message"].get("model")) for e in _assistant(driver)} - {DRIVER_MODEL})
    if models:
        found.add("driverModelMismatch", f"driver message.model {models} is not {DRIVER_MODEL}")
    _check_synthetics(found, errors)


def _check_synthetics(found: _Findings, errors: Sequence[Any]) -> None:
    other = sorted({str(error) for error in errors if error != RATE_LIMIT})
    if other:
        found.add("syntheticError", f"<synthetic> error {other}")
    if RATE_LIMIT in errors:
        found.add("rateLimit", "<synthetic> error 'rate_limit'")


def _check_run(
    found: _Findings, expected: Expected, spawn: Mapping[str, Any],
    matched: tuple[Mapping, Mapping] | None, meta: Mapping[str, Any],
    agent: Sequence[Mapping[str, Any]], errors: Sequence[Any], in_flight: bool,
) -> None:
    spawn_input = spawn["input"]
    users = [e for e in agent if e.get("type") == "user"]
    first = users[0].get("message", {}).get("content") if users else None
    if first != expected.spawn_prompt:
        found.add("spawnPromptMismatch", "the first user message is not the spawn prompt")

    overrides = [key for key in OVERRIDE_KEYS if key in spawn_input]
    if overrides:
        found.add("toolUseOverride", f"the Agent tool_use carries {overrides}")
    if spawn_input.get("subagent_type") != expected.agent_type:
        found.add("subagentTypeMismatch",
                  f"subagent_type {spawn_input.get('subagent_type')!r} is not the role's")

    if meta.get("agentType") != expected.agent_type:
        found.add("agentTypeMismatch", f"agentType {meta.get('agentType')!r} is not the role's")

    models = sorted({str(e["message"].get("model")) for e in _assistant(agent)}
                    - {expected.model})
    if models:
        found.add("modelMismatch", f"message.model {models} is not the pinned id")

    if "effort" not in spawn_input:
        efforts = _efforts(agent)
        wrong = sorted({str(effort) for effort in efforts if effort != expected.effort})
        if wrong:
            found.add("effortOverride", f"effort {wrong} is not the pin {expected.effort!r}")

    instructions = set(_instructions(agent))
    if instructions != set(expected.instructions):
        found.add("instructionsMismatch", "the instructions set is not the declared set")

    if _attachments(agent, "mcp_instructions_delta"):
        found.add("mcpInstructions", "an mcp_instructions_delta attachment")
    deferred = [entry.get("name") for record in _attachments(agent, "deferred_tools_record")
                for entry in record.get("entries") or [] if isinstance(entry, dict)]
    if any(isinstance(name, str) and name.startswith("mcp__") for name in deferred):
        found.add("mcpDeferredTool", "an mcp__ name in a deferred_tools_record")
    if any(isinstance(c.get("context"), dict) and "gitStatus" in c["context"]
           for c in _attachments(agent, "session_context")):
        found.add("gitStatus", "a gitStatus key in session_context")

    calls = _tool_uses(agent)
    allowed = _check_guard(found, expected, agent, calls, in_flight)

    if matched is not None and not errors and not _terminal(agent):
        found.add("maxTurns", "the run ended without a final reply")

    handbacks = [call for call in calls if call.get("name") == HANDBACK]
    if len(handbacks) > 1 or (handbacks and calls[-1] is not handbacks[0]):
        found.add("toolOutOfBounds", f"{len(handbacks)} {HANDBACK} calls, not one, last")
    _check_bounds(found, expected, allowed)


def _check_guard(found: _Findings, expected: Expected, entries: Sequence[Mapping[str, Any]],
                 calls: Sequence[Mapping[str, Any]], in_flight: bool) -> list[Mapping[str, Any]]:
    """Plan6 §3.4's guard rows over `calls`; returns the calls the guard allowed. With the spawn
    `in_flight`, a call with no `tool_result` is in flight too, and no row reads it."""
    results = _results(entries)
    guarded = {hook.get("toolUseID") for hook in _attachments(entries, "hook_success")
               if _is_guard_entry(hook, expected.guard_command)}
    attached = {e["attachment"].get("toolUseID") for e in entries
                if e.get("type") == "attachment" and isinstance(e.get("attachment"), dict)}
    allowed, rejected, invalid, unhooked = [], [], [], []
    for call in calls:
        call_id = call.get("id")
        entry, block = results.get(call_id, (None, None))
        if entry is None and in_flight:
            continue
        decision = _decision(entry) if entry is not None else None
        if decision == ("reject", "hook"):
            rejected.append(call_id)
        elif decision == ("accept", "hook") and call_id in guarded:
            allowed.append(call)
        elif call.get("name") == HANDBACK:
            continue  # exempt from hookMissing (plan6 §3.4 "Terminal shape")
        elif (entry is not None and decision is None and block.get("is_error") is True
              and call_id not in attached):
            invalid.append(call_id)
        else:
            unhooked.append(call_id)
    if rejected:
        found.add("guardRejected", f"the guard denied or failed on {rejected}")
    if invalid:
        found.add("inputInvalid", f"input validation refused {invalid} before any hook")
    if unhooked:
        found.add("hookMissing", f"the guard did not decide {unhooked}")
    return allowed


def _check_bounds(found: _Findings, expected: Expected,
                  allowed: Sequence[Mapping[str, Any]]) -> None:
    """Plan6 §3.4's bounds rows over the calls the guard allowed: outside the guard's bounds
    (§3.3) is `guardBreach`; inside them but not this run's own parts is counted."""
    folders = {"Read": os.path.dirname(expected.request_path),
               "Write": os.path.dirname(expected.answer_path)}
    request = os.path.realpath(expected.request_path)
    answer_part = _answer_part_pattern(expected.answer_path)
    breach, other = [], []
    for call in allowed:
        name = call.get("name")
        if name == HANDBACK:
            continue
        tool_input = call.get("input")
        raw = tool_input.get("file_path") if isinstance(tool_input, dict) else None
        path = _canonical(raw, call.get("_cwd"))
        if name not in folders or path is None or not _strictly_inside(path, folders[name]):
            breach.append(f"{name} {raw!r}")
        elif not (path == request if name == "Read" else answer_part.fullmatch(path)):
            other.append(f"{name} {raw!r}")
    if breach:
        found.add("guardBreach", f"the guard allowed calls beyond its bounds: {breach}")
    if other:
        found.add("pathOutOfBounds", f"paths beyond its request and answer parts: {other}")


def _terminal(agent: Sequence[Mapping[str, Any]]) -> bool:
    """Whether the last assistant entry is an `end_turn` reply, or exactly one hand-back
    followed only by its `tool_result` (plan6 §3.4 "Terminal shape")."""
    turns = [e for e in agent if e.get("type") in ("user", "assistant") and not _is_synthetic(e)]
    last = max((i for i, e in enumerate(turns) if e.get("type") == "assistant"), default=None)
    if last is None:
        return False
    uses = _blocks(turns[last], "tool_use")
    after = turns[last + 1:]
    if not after and not uses:
        return turns[last].get("message", {}).get("stop_reason") == "end_turn"
    return (len(uses) == 1 and uses[0].get("name") == HANDBACK and all(
        e.get("type") == "user" and _blocks(e, "tool_result") and all(
            b.get("tool_use_id") == uses[0].get("id") for b in _blocks(e, "tool_result"))
        for e in after))


def _efforts(agent: Sequence[Mapping[str, Any]]) -> list[Any]:
    requests = [e for e in agent if e.get("type") == "api-request"]
    if not requests:
        return [None] if _assistant(agent) else []
    return [_params(e).get("output_config", {}).get("effort") for e in requests]


def _params(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    params = entry.get("params")
    return params if isinstance(params, dict) else {}


def _instructions(agent: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    found = []
    for record in _attachments(agent, "instructions"):
        for file in record.get("files") or []:
            if isinstance(file, dict):
                content = file.get("content")
                text = content if isinstance(content, str) else json.dumps(content)
                found.append((str(file.get("path")),
                              hashlib.sha256(text.encode("utf-8")).hexdigest()))
    return found


def _facts(agent_id: str, agent: Sequence[Mapping[str, Any]]) -> RunFacts:
    conversation = [e for e in agent if e.get("type") in _CONVERSATION]
    models = [e["message"].get("model") for e in _assistant(agent)]
    contexts = [c.get("context") for c in _attachments(agent, "session_context")]
    requests = [_params(e) for e in agent if e.get("type") == "api-request"]
    return RunFacts(
        agent_id=agent_id,
        claude_code_version=conversation[0].get("version") if conversation else None,
        model=models[0] if models else None,
        instructions=tuple(sorted(set(_instructions(agent)))),
        session_context_sha256=hashlib.sha256(json.dumps(
            contexts[0], sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")).hexdigest() if contexts else None,
        thinking=requests[0].get("thinking") if requests else None,
        effort=requests[0].get("output_config", {}).get("effort") if requests else None,
        tokens=count_tokens(agent),
    )


_STATUS = {HALT: "void", REISSUE: "incomplete", WAIT: "usage-stop"}


def combine(result: Audit, outcome: Ingested | Void | None) -> Verdict:
    """Join the audit with `ingest_answer`'s outcome: the audit's class takes precedence (module
    docstring). `outcome` is `None` exactly when `result.ingests_answer` is false."""
    if isinstance(outcome, AlreadyIngested):
        raise ValueError("an AlreadyIngested outcome is a no-op, not a run to combine")
    if result.ingests_answer != (outcome is not None):
        raise ValueError(f"a {result.action} audit takes "
                         f"{'an ingest outcome' if result.ingests_answer else 'no ingest outcome'}")
    details = tuple(f.detail for f in result.findings)
    if outcome is None:
        never = result.void_class in (NEVER_SPAWNED, LAUNCH_FAILED)
        return Verdict(
            status=None if never else _STATUS[result.action],
            void_class=result.void_class, void_reason=result.reason, details=details,
            answer_sha256=None, halts=result.action == HALT, counts_attempt=False,
            reissues_under_new_name=result.action in (REISSUE, WAIT),
        )
    if result.action == COUNT:
        return Verdict("void", RUN_ATTRIBUTABLE, result.reason, details, outcome.answer_sha256,
                       halts=False, counts_attempt=True, reissues_under_new_name=False)
    if isinstance(outcome, Void):
        return Verdict("void", outcome.void_class, outcome.reason, outcome.details,
                       outcome.answer_sha256, halts=False, counts_attempt=True,
                       reissues_under_new_name=False)
    return Verdict("ingested", None, None, (), outcome.answer_sha256, halts=False,
                   counts_attempt=False, reissues_under_new_name=False)


def run_record(
    result: Audit, verdict: Verdict, *, run_id: str, role: str, batch_id: str, attempt: int,
    request_sha256: str,
) -> dict[str, Any] | None:
    """§3.7's `runs[]` entry, or `None` for a run with no located transcript."""
    facts = result.facts
    if facts is None:
        return None
    return {
        "runId": run_id, "role": role, "batchId": batch_id, "attempt": attempt,
        "status": verdict.status, "voidClass": verdict.void_class,
        "voidReason": verdict.void_reason, "claudeCodeVersion": facts.claude_code_version,
        "agentId": facts.agent_id, "model": facts.model, "requestSha256": request_sha256,
        "instructions": [{"path": path, "sha256": sha} for path, sha in facts.instructions],
        "sessionContextSha256": facts.session_context_sha256, "thinking": facts.thinking,
        "effort": facts.effort, "tokens": dict(facts.tokens),
    }
