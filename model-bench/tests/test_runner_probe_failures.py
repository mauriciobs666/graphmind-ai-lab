"""typed-decision-models U2.3 — a failed between-item residency probe (plan4 §3.10, §5 U2.3;
small-model-benchmarking plan §3.6 clause (iv) and its disposition-over-load precedence).

Every test drives the real `run_pack` against a stubbed `LMStudio`, one surface at a time: chat
(`_drive_single_call_items`), `tool-caller` (`_turn_timings`), `decisions` and `handoff`. Item 2's
between-item `residency()` raises; `probe()` then answers `"api-v0"` (the run completes, item 2 is
"not shown resident") or anything else (exit 3, nothing stored). Offline throughout; nothing is
written under `model-bench/results/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import test_runner as chat
import test_runner_decisions as dec
import test_runner_handoff as ho

from modelbench import cli
from modelbench.lmstudio import (
    LMStudioCallFailed,
    LMStudioCallTimeout,
    LMStudioUnreachable,
    ResidentModel,
)
from modelbench.runner import _PROBE_MESSAGES, RunRefused, run_pack


def _failure(kind: type[BaseException]) -> BaseException:
    """A fresh instance of one of the three ways `residency()` fails (`LMStudioUnreachable`,
    `LMStudioCallFailed`, a socket `TimeoutError`) — fresh, so no traceback leaks across tests."""
    if kind is LMStudioCallFailed:
        return LMStudioCallFailed("catalog GET returned 500", status=500, body=None)
    return kind("probe failed")


_PER_EXCEPTION = pytest.mark.parametrize(
    "kind", [LMStudioUnreachable, LMStudioCallFailed, TimeoutError], ids=lambda k: k.__name__
)


class _ProbeSequence:
    """`probe()` answers from a queue (default `"api-v0"`), counting its calls."""

    def _init_probes(self, probes: list[str] | None) -> None:
        self._probes = list(probes or [])
        self.probe_calls = 0

    def probe(self) -> str:
        self.probe_calls += 1
        return self._probes.pop(0) if self._probes else "api-v0"


class ChatStub(_ProbeSequence, chat.StubLMStudio):
    """`test_runner.StubLMStudio` whose residency snapshots may be exception instances (raised)."""

    def __init__(self, *, probes: list[str] | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self._init_probes(probes)

    def residency(self) -> list[ResidentModel]:
        self.residency_calls += 1
        snapshot = self._residency.pop(0) if self._residency else chat.resident()
        if isinstance(snapshot, BaseException):
            raise snapshot
        return snapshot


class DecisionsStub(_ProbeSequence, dec.DecisionStub):
    def __init__(self, *, probes: list[str] | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self._init_probes(probes)


class HandoffProbeStub(_ProbeSequence, ho.HandoffStub):
    def __init__(self, *, probes: list[str] | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self._init_probes(probes)


# --- chat (`_drive_single_call_items`) -----------------------------------------------------------


_CHAT_ITEMS = [{"id": x, "itemId": x} for x in ("a", "b", "c")]


def _chat_pack() -> chat.FakePack:
    return chat.FakePack(
        role="guard-judge",
        items=list(_CHAT_ITEMS),
        manifest={"environment": {"requires": ["lmstudio-chat"]}},
        prompt_cfg=chat.make_prompt_cfg(maxIterationsPerTurn=None),
    )


def _chat_run(tmp_path: Path, monkeypatch, lms: ChatStub):
    monkeypatch.setattr("modelbench.runner._load_item_scorer", lambda pack: chat.FakeItemScorer())
    chat._host_json(tmp_path)
    return run_pack(_chat_pack(), chat._cfg(), lmstudio=lms, root=tmp_path)


def _chat_stub(failure: BaseException, *, item_2: Any = None, probes=None) -> ChatStub:
    ok = chat.resident()
    # start, baseline, item 1, item 2 (raises), item 3, step 10
    return ChatStub(
        catalog_result=[chat.model_info()],
        chat_responses=[
            chat.chat_result_with_stats(),
            item_2 if item_2 is not None else chat.chat_result_with_stats(),
            chat.chat_result_with_stats(),
        ],
        residency_sequence=[ok, ok, ok, failure, ok, ok],
        probes=probes,
    )


@_PER_EXCEPTION
def test_chat_a_completed_item_after_a_failed_probe_is_withheld_for_load(
    tmp_path, monkeypatch, capsys, kind
):
    lms = _chat_stub(_failure(kind))
    run, _ = _chat_run(tmp_path, monkeypatch, lms)
    assert [i.timing.withheldFor for i in run.items] == [None, "load", None]
    assert lms.probe_calls == 2  # step 2, then the re-probe
    assert len(lms.chat_calls) == 3
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1
    assert "'b'" in lines[0] and kind.__name__ in lines[0]


@_PER_EXCEPTION
@pytest.mark.parametrize(
    ("call_failure", "withheld"),
    [
        (LMStudioCallTimeout("timed out"), "timeout"),
        (LMStudioCallFailed("dropped", status=None, body=None), "no_response"),
    ],
    ids=["timeout", "no_response"],
)
def test_chat_a_failed_item_after_a_failed_probe_keeps_its_own_disposition(
    tmp_path, monkeypatch, kind, call_failure, withheld
):
    """SMB §3.6's precedence: the item's own failure wins, and no load producer runs on it."""
    run, _ = _chat_run(tmp_path, monkeypatch, _chat_stub(_failure(kind), item_2=call_failure))
    assert [i.timing.withheldFor for i in run.items] == [None, withheld, None]


@pytest.mark.parametrize("answer", ["unreachable", "v1-only"])
def test_chat_a_failed_probe_with_the_server_gone_refuses_with_exit_3(
    tmp_path, monkeypatch, answer
):
    lms = _chat_stub(LMStudioUnreachable("gone"), probes=["api-v0", answer])
    with pytest.raises(RunRefused) as caught:
        _chat_run(tmp_path, monkeypatch, lms)
    assert caught.value.exitCode == 3
    assert str(caught.value) == _PROBE_MESSAGES[answer]
    assert len(lms.chat_calls) == 1  # item 2's call is never made


def test_chat_any_other_exception_from_the_snapshot_propagates_without_a_re_probe(
    tmp_path, monkeypatch
):
    """Impl review P20-n1: the catch is exactly the three probe failures. A defect raising from
    `residency()` stays a traceback, never a re-probe that turns it into a `"load"` withholding."""
    lms = _chat_stub(RuntimeError("a defect, not a probe failure"))
    with pytest.raises(RuntimeError, match="a defect"):
        _chat_run(tmp_path, monkeypatch, lms)
    assert lms.probe_calls == 1  # step 2 only


# --- tool-caller (`_turn_timings`) ---------------------------------------------------------------


def _tool_caller_run(tmp_path: Path, monkeypatch, lms: ChatStub):
    monkeypatch.setattr(
        "modelbench.runner._load_conversation_scorer", lambda pack: chat.FakeConversationScorer()
    )
    chat._host_json(tmp_path)
    pack = chat._tool_caller_pack(scripts=[chat.script(s) for s in ("s1", "s2", "s3")])
    pack.manifest["environment"] = {"requires": ["lmstudio-chat"]}
    return run_pack(pack, chat._cfg(), lmstudio=lms, root=tmp_path)


def _tool_caller_stub(failure: BaseException, *, conv_2: Any = None, probes=None) -> ChatStub:
    ok = chat.resident()
    # start, baseline, after s1 (raises: guards s2), after s2, after s3, step 10
    return ChatStub(
        catalog_result=[chat.model_info()],
        chat_responses=[
            chat.chat_result_with_stats(),
            conv_2 if conv_2 is not None else chat.chat_result_with_stats(),
            chat.chat_result_with_stats(),
        ],
        residency_sequence=[ok, ok, failure, ok, ok, ok],
        probes=probes,
    )


@_PER_EXCEPTION
def test_turn_timings_a_completed_turn_after_a_failed_probe_is_withheld_for_load(
    tmp_path, monkeypatch, capsys, kind
):
    lms = _tool_caller_stub(_failure(kind))
    run, _ = _tool_caller_run(tmp_path, monkeypatch, lms)
    assert [i.timing.withheldFor for i in run.items] == [None, "load", None]
    assert lms.probe_calls == 2
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1
    assert "'s1'" in lines[0] and kind.__name__ in lines[0]


@_PER_EXCEPTION
def test_turn_timings_a_timed_out_turn_after_a_failed_probe_stays_timeout(
    tmp_path, monkeypatch, kind
):
    lms = _tool_caller_stub(_failure(kind), conv_2=LMStudioCallTimeout("timed out"))
    run, _ = _tool_caller_run(tmp_path, monkeypatch, lms)
    assert [i.timing.withheldFor for i in run.items] == [None, "timeout", None]


def test_turn_timings_a_failed_probe_after_the_last_conversation_changes_no_item(
    tmp_path, monkeypatch, capsys
):
    """Impl review P20-n2: the snapshot after the last conversation guards nothing, so no item
    moves and the one stderr line claims no item."""
    ok = chat.resident()
    # start, baseline, after s1, after s2, after s3 (raises: guards nothing), step 10
    lms = ChatStub(
        catalog_result=[chat.model_info()],
        chat_responses=[chat.chat_result_with_stats() for _ in range(3)],
        residency_sequence=[ok, ok, ok, ok, LMStudioUnreachable("probe failed"), ok],
    )
    run, _ = _tool_caller_run(tmp_path, monkeypatch, lms)
    assert [i.timing.withheldFor for i in run.items] == [None, None, None]
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1
    assert "'s3'" in lines[0] and "that item" not in lines[0]


def test_turn_timings_a_failed_probe_with_the_server_gone_refuses_with_exit_3(
    tmp_path, monkeypatch
):
    lms = _tool_caller_stub(LMStudioUnreachable("gone"), probes=["api-v0", "unreachable"])
    with pytest.raises(RunRefused) as caught:
        _tool_caller_run(tmp_path, monkeypatch, lms)
    assert caught.value.exitCode == 3
    assert str(caught.value) == _PROBE_MESSAGES["unreachable"]
    assert len(lms.chat_calls) == 1


# --- decisions (`_drive_decision_items`) ---------------------------------------------------------


def _decisions_sequence(failure: BaseException) -> list[Any]:
    laya = list(dec._LAYA_RESIDENT)
    # start, baseline, items 1-4 (item 2 raises), the load rule's probe, step 10
    return [laya, laya, laya, failure, laya, laya, laya, laya]


@_PER_EXCEPTION
@pytest.mark.parametrize("item_2_call", ["succeeds", "fails"])
def test_decisions_item_2_after_a_failed_probe_is_load_and_the_run_assumed(
    tmp_path, capsys, kind, item_2_call
):
    dec._workspace(tmp_path)
    responses: list[Any] = dec._four(0.95, 0.1, 0.85, 0.9)
    if item_2_call == "fails":
        responses[1] = LMStudioCallTimeout("timed out")
    lms = DecisionsStub(
        decide_responses=responses, residency_sequence=_decisions_sequence(_failure(kind))
    )
    run, _ = dec._run(tmp_path, lms)
    assert [i.timing.withheldFor for i in run.items] == [None, "load", None, None]
    assert [i.itemId for i in run.items] == list(dec._ITEM_IDS)
    assert run.basis == "assumed"
    assert lms.probe_calls == 2
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1
    assert repr(dec._ITEM_IDS[1]) in lines[0] and kind.__name__ in lines[0]


def test_decisions_a_failed_probe_with_the_server_gone_refuses_with_exit_3(tmp_path):
    dec._workspace(tmp_path)
    lms = DecisionsStub(
        decide_responses=dec._four(0.95, 0.1, 0.85, 0.9),
        residency_sequence=_decisions_sequence(LMStudioUnreachable("gone")),
        probes=["api-v0", "unreachable"],
    )
    with pytest.raises(RunRefused) as caught:
        dec._run(tmp_path, lms)
    assert caught.value.exitCode == 3
    assert str(caught.value) == _PROBE_MESSAGES["unreachable"]
    assert len(lms.decide_calls) == 1


def test_cli_run_with_the_server_gone_at_a_between_item_probe_exits_3_storing_nothing(
    tmp_path, monkeypatch, capsys
):
    """Only `cli` stores (impl review P9-n1), so "stores nothing" is asserted through `cli.main`."""
    dec._workspace(tmp_path, contract_valid=True)
    lms = DecisionsStub(
        decide_responses=dec._four(0.95, 0.1, 0.85, 0.9),
        residency_sequence=_decisions_sequence(LMStudioUnreachable("gone")),
        probes=["api-v0", "unreachable"],
    )
    monkeypatch.setattr(cli, "LMStudio", lambda base_url: lms)
    code = cli.main(
        ["run", "--root", str(tmp_path), "--pack", dec._PACK_ID, "--candidate", dec._CANDIDATE_ID]
    )
    err = capsys.readouterr().err
    assert code == 3, err
    assert _PROBE_MESSAGES["unreachable"] in err
    assert not (tmp_path / "results").exists()


# --- handoff (`_drive_handoff_items`) ------------------------------------------------------------


def _handoff_sequence(failure: BaseException) -> list[Any]:
    both = ho._resident("laya", ho.QWEN)
    # start, baseline, items 1-4 (item 2 raises), the load rule's probe, step 10
    return [both, both, both, failure, both, both, both, both]


@_PER_EXCEPTION
@pytest.mark.parametrize("item_2_call", ["succeeds", "fails"])
def test_handoff_item_2_after_a_failed_probe_is_load_and_the_run_assumed(
    tmp_path, capsys, kind, item_2_call
):
    ho._handoff_workspace(tmp_path)
    responses: list[Any] = list(ho._MIXED)  # item 2 decides (0.05), items 3 and 4 hand off
    if item_2_call == "fails":
        responses[1] = LMStudioCallTimeout("timed out")
    lms = HandoffProbeStub(
        decide_responses=responses, residency_sequence=_handoff_sequence(_failure(kind))
    )
    run, _ = ho._run(tmp_path, lms)
    assert [i.timing.withheldFor for i in run.items] == [None, "load", None, None]
    assert [i.detail["decidedBy"] for i in run.items[2:]] == ["handoff", "handoff"]
    assert run.basis == "assumed"
    assert lms.probe_calls == 2
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1
    assert repr(dec._ITEM_IDS[1]) in lines[0] and kind.__name__ in lines[0]


def test_handoff_a_failed_probe_with_the_server_gone_refuses_with_exit_3(tmp_path):
    ho._handoff_workspace(tmp_path)
    lms = HandoffProbeStub(
        decide_responses=list(ho._MIXED),
        residency_sequence=_handoff_sequence(LMStudioUnreachable("gone")),
        probes=["api-v0", "unreachable"],
    )
    with pytest.raises(RunRefused) as caught:
        ho._run(tmp_path, lms)
    assert caught.value.exitCode == 3
    assert str(caught.value) == _PROBE_MESSAGES["unreachable"]
    assert len(lms.decide_calls) == 1
