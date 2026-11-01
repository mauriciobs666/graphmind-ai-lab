"""typed-decision-models U2.2 — the `handoff` surface: a decision-model candidate with a hand-off
chat model (plan v1.3 §3.3, §4 U2, §5 U2; plan2 §3.3–§3.4, §4.1 U2, §5 U2; plan3 §3.2, §3.3's
`handoff` rows, §3.6, §3.7, §5 U2).

Offline throughout, on `test_runner_decisions`' fixture pack and candidate workspace, re-shaped to
a hand-off candidate (a 0.1–0.9 band, `qwen/qwen3-4b-2507` — the fixture pack's incumbent — as
its hand-off model). `HandoffStub` extends that module's `DecisionStub` with chat calls, residency
derived from what is loaded, and one instance config per model. Its `decide` and `chat` advance
one shared fake clock (`_StubClock`) by each call's own duration, so an item's `wallClockMs` is
exactly the calls the item clock spans (impl review P9-M1). Nothing here writes under
`model-bench/results/`.

The new-ability (`fail-closed`) rows have no role to run on yet — `HANDOFF_FAILURE_BY_ROLE` holds
the guard alone until U4 — so those tests patch the runner's copy of the table to make the guard
fixture `fail-closed`, and use a scorer that accepts `failed_call="handoff"`, which
`classification.score_answer` (the guard's) refuses by design. That is plan2 §5 U2's "join-in
fixture" in the only shape available before U4.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
import test_runner_decisions as base

from modelbench.lmstudio import (
    WARMUP_DECISION_INPUT,
    ChatResult,
    LMStudioCallFailed,
    LMStudioCallTimeout,
    ResidentModel,
)
from modelbench.packs import load_pack
from modelbench.results import ItemResult, ItemTiming, load_history, store
from modelbench.runner import RunRefused, _build_fingerprint, latency_block
from modelbench.scoring import classification

QWEN = "qwen/qwen3-4b-2507"
#: A hand-off band: p >= 0.9 yes, p < 0.1 no, anything between is handed off.
_BAND = {"kind": "predicate-band", "yesAtOrAbove": 0.9, "noBelow": 0.1}
#: qwen's own `loaded_instances[0].config` — deliberately unlike laya's, so a read made under the
#: wrong model id records a visibly different `decisionInstanceConfig`.
_QWEN_CONFIG = {"context_length": 8192, "flash_attention": True, "offload_kv_cache_to_gpu": True}
#: Chat stats whose per-call gap (3000 − (200 + 500) = 2300 ms) is past `-ml` §11.5.1's 1000 ms
#: threshold: if a gap check ran on this surface, the item would be withheld for `"load"`.
_GAPPY_STATS = {"time_to_first_token": 0.2, "generation_time": 0.5, "tokens_per_second": 40.0}


def _reply(decision: bool, *, wall_clock_ms: float = 3000.0) -> ChatResult:
    return ChatResult(
        message={"content": json.dumps({"decision": decision})},
        tool_calls=(),
        toolCallForm="prose",
        stats=dict(_GAPPY_STATS),
        model_info=None,
        runtime={"name": "llama.cpp", "version": "1.52.0"},
        usage={"prompt_tokens": 300, "completion_tokens": 8, "total_tokens": 308},
        wallClockMs=wall_clock_ms,
    )


#: How long a raising call takes on `_StubClock`: a failed `decide` (an over-limit 500, a timeout)
#: and a failed hand-off `chat`. Distinct from each other and from the successful calls' 12 ms and
#: 3000 ms, so an exact sum names which calls the item clock spanned.
_FAILED_DECIDE_MS = 40.0
_FAILED_CHAT_MS = 25.0


class _StubClock:
    """The item clock (seconds): reading it never advances it; the stub's calls do, each by its
    own duration — a returned call by its `wallClockMs`, a raising one by `_FAILED_*_MS`."""

    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t

    def advance(self, ms: float) -> None:
        self.t += ms / 1000.0


def _resident(*ids: str) -> list[ResidentModel]:
    return [ResidentModel(id=m, state="loaded") for m in ids]


class HandoffStub(base.DecisionStub):
    """`DecisionStub` plus the hand-off model.

    - `chat` pops `chat_responses` in call order (an exception instance is raised); once empty it
      answers `{"decision": false}`. Each call is logged with its messages, model, sampling and
      timeout.
    - `residency`, when no `residency_sequence` entry is queued, reports exactly what is loaded,
      so a load or an eviction shows in the next snapshot.
    - `loaded_instances` answers with each model's own config.
    - `evict_on_chat_warm_up` unloads laya when qwen's chat warm-up runs (a JIT eviction).
    - `chat_runtime` is what the chat warm-up reports as `runtime`.
    - `clock`, when set, is advanced by every `decide` and `chat` call's own duration."""

    def __init__(
        self,
        *,
        chat_responses: list[Any] | None = None,
        evict_on_chat_warm_up: bool = False,
        chat_runtime: Any = None,
        chat_warm_up_failure: BaseException | None = None,
        initially_loaded: tuple[str, ...] = ("laya", QWEN),
        **kw: Any,
    ) -> None:
        super().__init__(initially_loaded=initially_loaded, **kw)
        self._chat = list(chat_responses or [])
        self.evict_on_chat_warm_up = evict_on_chat_warm_up
        self.chat_runtime = (
            chat_runtime if chat_runtime is not None else {"name": "llama.cpp", "version": "1.52.0"}
        )
        self.chat_warm_up_failure = chat_warm_up_failure
        self.instance_configs = {"laya": base._LAYA_CONFIG, QWEN: _QWEN_CONFIG}
        self.clock: _StubClock | None = None

    def _advance(self, ms: float) -> None:
        if self.clock is not None:
            self.clock.advance(ms)

    def decide(self, input, questions, *, model, timeout_s):
        try:
            result = super().decide(input, questions, model=model, timeout_s=timeout_s)
        except (LMStudioCallTimeout, LMStudioCallFailed):
            self._advance(_FAILED_DECIDE_MS)
            raise
        self._advance(result.wallClockMs)
        return result

    def residency(self) -> list[ResidentModel]:
        self.events.append("residency")
        if self._residency:
            snapshot = self._residency.pop(0)
            if isinstance(snapshot, BaseException):
                raise snapshot
            return snapshot
        return _resident(*sorted(self.loaded))

    def loaded_instances(self, model_key):
        self.order.append(("loaded_instances", model_key))
        if model_key not in self.loaded:
            return ()
        return ({"id": model_key, "config": self.instance_configs[model_key]},)

    def warm_up(self, model_key, *, call_surface, **kw):
        if call_surface == "chat" and self.chat_warm_up_failure is not None:
            self.order.append(("warm_up", model_key))
            raise self.chat_warm_up_failure
        result = super().warm_up(model_key, call_surface=call_surface, **kw)
        if call_surface == "chat":
            result = type(result)(
                wallClockMs=result.wallClockMs,
                wasResidentBefore=result.wasResidentBefore,
                runtime=self.chat_runtime,
                stats=None,
            )
            if self.evict_on_chat_warm_up:
                self.loaded.discard("laya")
        return result

    def chat(self, messages, *, model, temperature, max_tokens, timeout_s, tools=None):
        self.chat_calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout_s": timeout_s,
            }
        )
        self.events.append(("chat", model))
        response = self._chat.pop(0) if self._chat else _reply(False)
        if isinstance(response, BaseException):
            self._advance(_FAILED_CHAT_MS)
            raise response
        self._advance(response.wallClockMs)
        return response


def _handoff_workspace(root: Path, **candidate: Any) -> Path:
    candidate.setdefault("handoffModel", QWEN)
    candidate.setdefault("policy", dict(_BAND))
    return base._workspace(root, **candidate)


def _run(root: Path, lms: HandoffStub, **cfg: Any):
    """`run_pack` on a `_StubClock` that `lms`'s calls advance: an item's `wallClockMs` is the sum
    of the calls inside its item clock."""
    lms.clock = _StubClock()
    return base._run(root, lms, clock=lms.clock, **cfg)


def _over_limit() -> LMStudioCallFailed:
    return base._failure("body-document-at-500")


def _items(root: Path) -> list[dict[str, Any]]:
    return list(load_pack(root / "packs" / base._PACK_ID).iter_items())


_metric = base._metric


def _has_metric(run, name: str) -> bool:
    return any(m.name == name for m in run.aggregates.perClass)


def _counts(metric) -> tuple[int, int]:
    return metric.successes, metric.n


class FailClosedScorer:
    """`classification`'s hooks, plus a `score_answer` that accepts `failed_call="handoff"` and
    records the keywords it was handed. Stands in for U4's fail-closed scorer, with the guard
    fixture's table row patched to `fail-closed` (module docstring)."""

    decision_input = staticmethod(classification.decision_input)
    answer_from_chat = staticmethod(classification.answer_from_chat)
    build_messages = staticmethod(classification.build_messages)
    score_item = staticmethod(classification.score_item)
    aggregate = staticmethod(classification.aggregate)

    def __init__(self) -> None:
        self.handoff_failures: list[dict[str, Any]] = []

    def score_answer(self, item_input, answer, outcome, timing, *, pack, decided_by,
                     probability, failed_call, reason, input_tokens):
        if failed_call != "handoff":
            return classification.score_answer(
                item_input, answer, outcome, timing, pack=pack, decided_by=decided_by,
                probability=probability, failed_call=failed_call, reason=reason,
                input_tokens=input_tokens,
            )
        self.handoff_failures.append(
            {"answer": answer, "outcome": outcome, "decided_by": decided_by,
             "probability": probability, "reason": reason, "input_tokens": input_tokens}
        )
        detail = {"path": item_input["path"], "tier": item_input["tier"], "failedCall": "handoff"}
        if reason is not None:
            detail["reason"] = reason
        return ItemResult(
            itemId=item_input["itemId"], pairingKey=(item_input["itemId"],), outcome=outcome,
            scoreable={}, counts={}, timing=timing, detail=detail,
        )


@pytest.fixture
def fail_closed(monkeypatch) -> FailClosedScorer:
    scorer = FailClosedScorer()
    table = MappingProxyType({"guard-judge": "fail-closed"})
    monkeypatch.setattr("modelbench.runner.HANDOFF_FAILURE_BY_ROLE", table)
    monkeypatch.setattr("modelbench.runner._load_item_scorer", lambda pack: scorer)
    return scorer


# ==================================================================================================
# 1. v1.3 §3.3's `handoff` refusals (exit 4, before any call)
# ==================================================================================================


def _refused(root: Path, lms: HandoffStub) -> RunRefused:
    with pytest.raises(RunRefused) as caught:
        _run(root, lms)
    assert lms.warm_up_calls == [] and lms.decide_calls == [] and lms.chat_calls == []
    assert lms.load_calls == []
    return caught.value


def test_a_hand_off_candidate_on_a_pack_without_the_chat_token_refuses(tmp_path):
    def embeddings_only(manifest):
        manifest["environment"]["requires"] = ["lmstudio-embeddings", "lmstudio-decisions"]

    _handoff_workspace(tmp_path, manifest_edit=embeddings_only)
    exc = _refused(tmp_path, HandoffStub())
    assert exc.exitCode == 4 and "'lmstudio-chat'" in str(exc)


def test_a_hand_off_model_whose_catalog_type_is_not_llm_or_vlm_refuses(tmp_path):
    qwen = base._qwen()
    not_a_chat_model = type(qwen)(**{**qwen.__dict__, "type": "embeddings"})
    _handoff_workspace(tmp_path)
    exc = _refused(tmp_path, HandoffStub(catalog_before=[base._laya(), not_a_chat_model]))
    assert exc.exitCode == 4 and "catalog type 'embeddings', not 'llm' or 'vlm'" in str(exc)


def test_a_vlm_hand_off_model_is_accepted(tmp_path):
    qwen = base._qwen()
    vlm = type(qwen)(**{**qwen.__dict__, "type": "vlm"})
    _handoff_workspace(tmp_path)
    run, _ = _run(tmp_path, HandoffStub(catalog_before=[base._laya(), vlm]))
    assert run.fingerprint.get("handoffModelType") == "vlm"


def test_a_hand_off_model_absent_from_the_catalog_refuses(tmp_path):
    _handoff_workspace(tmp_path)
    exc = _refused(tmp_path, HandoffStub(catalog_before=[base._laya()]))
    assert exc.exitCode == 4 and repr(QWEN) in str(exc)


def test_a_speed_up_hand_off_model_that_is_not_the_incumbent_refuses(tmp_path):
    """v1.3 §3.3 / FR-3: on a speed-up point the hand-off model must be the incumbent; anything
    else goes to the existing judge."""
    qwen = base._qwen()
    other = type(qwen)(**{**qwen.__dict__, "id": "other/llm-7b"})
    _handoff_workspace(tmp_path, handoffModel="other/llm-7b")
    exc = _refused(tmp_path, HandoffStub(catalog_before=[base._laya(), qwen, other]))
    assert exc.exitCode == 4 and "not the speed-up point's incumbent" in str(exc)


def test_a_scorer_without_the_hand_off_parse_refuses(tmp_path, monkeypatch):
    class NoHandoffParse:
        decision_input = staticmethod(classification.decision_input)
        score_answer = staticmethod(classification.score_answer)
        score_item = staticmethod(classification.score_item)
        aggregate = staticmethod(classification.aggregate)

    monkeypatch.setattr("modelbench.runner._load_item_scorer", lambda pack: NoHandoffParse())
    _handoff_workspace(tmp_path)
    exc = _refused(tmp_path, HandoffStub())
    assert exc.exitCode == 4 and "no answer_from_chat hook" in str(exc)


def test_a_hand_off_candidate_with_an_overlapping_fit_manifest_refuses_with_exit_4(tmp_path):
    """v1.3 §4 U2's done condition: an overlapping fit manifest refuses (AC-3)."""
    from modelbench.decisions import input_hash

    pack_dir = _handoff_workspace(tmp_path)
    first = json.loads((pack_dir / "items.jsonl").read_text().splitlines()[0])
    leaked = tuple(sorted((*base._MANIFEST_HASHES, input_hash(first, "guard-judge"))))
    manifest = base._manifest_bytes(leaked)
    import hashlib

    body = json.loads((tmp_path / "candidates" / f"{base._CANDIDATE_ID}.json").read_text())
    body["fitManifest"].update(sha256=hashlib.sha256(manifest).hexdigest(), size=len(leaked))
    (tmp_path / "candidates" / f"{base._CANDIDATE_ID}.json").write_text(json.dumps(body))
    (tmp_path / "candidates" / f"{base._CANDIDATE_ID}.fit.txt").write_bytes(manifest)
    exc = _refused(tmp_path, HandoffStub())
    assert exc.exitCode == 4 and "AC-3" in str(exc)


# ==================================================================================================
# 2. The per-item flow (v1.3 §3.3 "Per-item flow, handoff surface"; v1.3 §5 U2)
# ==================================================================================================

#: Items in file order: fx-cs-01 decided yes (0.95), fx-cs-02 decided no (0.05), fx-ca-01 and
#: fx-ca-02 handed off (0.5, 0.3). The two hand-offs answer advance and suspend.
_MIXED = base._four(0.95, 0.05, 0.5, 0.3)


def test_a_mixed_run_decides_two_items_and_hands_off_two(tmp_path):
    _handoff_workspace(tmp_path)
    lms = HandoffStub(decide_responses=list(_MIXED), chat_responses=[_reply(True), _reply(False)])
    run, _ = _run(tmp_path, lms)
    details = [
        (i.detail["decidedBy"], i.detail["probability"], i.detail["inputTokens"])
        for i in run.items
    ]
    assert details == [
        ("decision-model", 0.95, 40),
        ("decision-model", 0.05, 40),
        ("handoff", 0.5, 40),
        ("handoff", 0.3, 40),
    ]
    assert all("failedCall" not in i.detail and "reason" not in i.detail for i in run.items)
    counts = {i.itemId: i.counts for i in run.items}
    # U3b.1: per-kind verdict metric plus the pooled one, per eval item (v1.3 §3.5).
    assert counts == {
        # decided yes on a clear suspend
        "fx-cs-01": {"falseAdvanceRateByUnderstanding": 1, "falseAdvanceRate": 1},
        "fx-cs-02": {"falseAdvanceRateByTurns": 0, "falseAdvanceRate": 0},
        # the hand-off LLM advanced
        "fx-ca-01": {"falseSuspendRateByUnderstanding": 0, "falseSuspendRate": 0},
        # the hand-off LLM suspended
        "fx-ca-02": {"falseSuspendRateByTurns": 1, "falseSuspendRate": 1},
    }
    assert [len(lms.decide_calls), len(lms.chat_calls)] == [4 + 1, 2]  # + the re-check


def test_a_hand_off_reply_with_a_true_string_is_a_suspend_on_a_decision_point_pack(tmp_path):
    """v1.3 §5 U3b at the hand-off seam (impl review P14-m1): `_drive_handoff_items` hands the
    parse its decision-point pack, so `"decision": "true"` is falkor-chat's suspend, not the old
    pack's loose advance. Both hand-offs are `clear_advance` items, so each is a false suspend."""
    _handoff_workspace(tmp_path)
    string_true = dataclasses.replace(_reply(True), message={"content": '{"decision": "true"}'})
    lms = HandoffStub(decide_responses=list(_MIXED), chat_responses=[string_true, string_true])
    run, _ = _run(tmp_path, lms)
    handed = {i.itemId: i.counts for i in run.items if i.detail["decidedBy"] == "handoff"}
    assert handed == {
        "fx-ca-01": {"falseSuspendRateByUnderstanding": 1, "falseSuspendRate": 1},
        "fx-ca-02": {"falseSuspendRateByTurns": 1, "falseSuspendRate": 1},
    }
    assert all(i.outcome == "pass" for i in run.items)


def test_a_handed_off_item_has_two_calls_and_a_decided_one_has_one(tmp_path):
    """v1.3 §5 U2: two calls, item wall clock >= the sum of the call clocks; one call when
    decided. The chat call's timing carries its stats (v1.3 §3.3 step 4). Plan3 §3.3 / note
    §Q4.3: the item clock spans the decision call **and** the hand-off — FR-2's `y_i` — so a
    hand-off is exactly 12 + 3000 ms on the stub clock (impl review P9-M1)."""
    _handoff_workspace(tmp_path)
    lms = HandoffStub(decide_responses=list(_MIXED), chat_responses=[_reply(True), _reply(False)])
    run, _ = _run(tmp_path, lms)
    decided, handed = run.items[0], run.items[2]
    assert [c.wallClockMs for c in decided.timing.calls] == [12.0]
    assert decided.timing.calls[0].ttftMs is None
    assert [c.wallClockMs for c in handed.timing.calls] == [12.0, 3000.0]
    chat_call = handed.timing.calls[1]
    assert (chat_call.ttftMs, chat_call.generationMs, chat_call.promptTokens) == (200.0, 500.0, 300)
    assert [i.timing.wallClockMs for i in run.items] == pytest.approx(
        [12.0, 12.0, 3012.0, 3012.0]
    )
    for item in run.items:
        assert item.timing.wallClockMs >= sum(c.wallClockMs for c in item.timing.calls)
        assert item.timing.withheldFor is None


def test_each_call_carries_its_own_model_input_and_timeout(tmp_path):
    """The decide call: the rendered input, laya, `decisionTimeoutSeconds` (2.0). The hand-off
    chat call: the scorer's `build_messages`, qwen, the pack's sampling, and the pinned 180 s —
    never `decisionTimeoutSeconds` (plan3 §3.2)."""
    _handoff_workspace(tmp_path)
    lms = HandoffStub(decide_responses=list(_MIXED))
    _run(tmp_path, lms)
    items = _items(tmp_path)
    scored_decides = lms.decide_calls[:4]
    assert [c["input"] for c in scored_decides] == [
        classification.decision_input(i, pack=None) for i in items
    ]
    assert {(c["model"], c["timeout_s"]) for c in scored_decides} == {("laya", 2.0)}
    assert [c["messages"] for c in lms.chat_calls] == [
        classification.build_messages(i, pack=load_pack(tmp_path / "packs" / base._PACK_ID))
        for i in items[2:]
    ]
    assert {(c["model"], c["temperature"], c["max_tokens"], c["timeout_s"])
            for c in lms.chat_calls} == {(QWEN, 0.0, 256, 180.0)}


def test_the_hand_off_chat_call_never_gets_the_decision_timeout_on_either_path(tmp_path):
    """Plan3 §3.2 / §5 U2: a policy hand-off and an over-limit hand-off both chat at the run's
    chat budget (180 s), and every decide call at `decisionTimeoutSeconds`."""
    _handoff_workspace(tmp_path, decisionTimeoutSeconds=7.5)
    lms = HandoffStub(decide_responses=[base._decided(0.5), _over_limit(), *base._four(0.95, 0.95)])
    _run(tmp_path, lms)
    assert [c["timeout_s"] for c in lms.chat_calls] == [180.0, 180.0]
    assert {c["timeout_s"] for c in lms.decide_calls} == {7.5}


def test_an_explicit_180_request_timeout_is_what_the_hand_off_uses(tmp_path):
    _handoff_workspace(tmp_path)
    lms = HandoffStub(decide_responses=list(_MIXED))
    run, _ = _run(tmp_path, lms, requestTimeoutSeconds=180.0)
    assert run.requestTimeoutSeconds == 180.0
    assert [c["timeout_s"] for c in lms.chat_calls] == [180.0, 180.0]


def test_a_request_timeout_other_than_180_refuses_a_hand_off_run_up_front(tmp_path):
    _handoff_workspace(tmp_path)
    lms = HandoffStub()
    with pytest.raises(RunRefused) as caught:
        _run(tmp_path, lms, requestTimeoutSeconds=120.0)
    assert caught.value.exitCode == 4
    assert lms.chat_calls == [] and lms.warm_up_calls == []


@pytest.mark.parametrize("absent", ["laya", QWEN])
def test_an_item_after_a_snapshot_missing_either_model_is_withheld_for_load(tmp_path, absent):
    """v1.3 §3.3 / §5 U2: the between-item probe must show **both** models resident."""
    _handoff_workspace(tmp_path)
    both = _resident("laya", QWEN)
    one = [m for m in both if m.id != absent]
    # start, baseline, item 1..4, the load rule's probe, step 10
    sequence = [both, both, both, one, both, both, both, both]
    lms = HandoffStub(decide_responses=list(_MIXED), residency_sequence=sequence)
    run, _ = _run(tmp_path, lms)
    assert [i.timing.withheldFor for i in run.items] == [None, "load", None, None]
    # item 2 is load-withheld and decided — attempted, so in n; items 3 and 4 hand off (P9-m2)
    assert _counts(_metric(run, "handoffShare")) == (2, 4)
    assert run.latency.latencyWithheldForLoad == 1
    assert run.basis == "assumed"  # plan2 §3.4: a load-withheld item means "not one load"


def test_a_load_withheld_hand_off_keeps_its_two_calls_and_the_load_withholding(tmp_path):
    _handoff_workspace(tmp_path)
    both = _resident("laya", QWEN)
    sequence = [both, both, both, both, _resident("laya"), both, both, both]
    lms = HandoffStub(decide_responses=list(_MIXED), residency_sequence=sequence)
    run, _ = _run(tmp_path, lms)
    handed = run.items[2]
    assert handed.detail["decidedBy"] == "handoff"
    assert (len(handed.timing.calls), handed.timing.withheldFor) == (2, "load")


# ==================================================================================================
# 3. Plan3 §3.3's over-limit `handoff` rows, one column each, field by field and in the counts
#    (`handoffShare`, `overLimit`, `handoffCallFailures`), plus `decisionCallFailures`
# ==================================================================================================

#: Item 2 (fx-cs-02, a clear suspend) is over the limit; the other three are decided.
_OVER_LIMIT_SECOND = [base._decided(0.95), _over_limit(), base._decided(0.05), base._decided(0.95)]


def test_over_limit_hand_off_succeeds_column(tmp_path):
    """Plan3 §3.3, "Hand-off succeeds": `reason=over-limit`, `decidedBy=handoff`, no
    `failedCall`, `calls=(chat,)`, the item clock over the failed call and the hand-off,
    `withheldFor=None` (no gap check, though the lone chat call's gap is past the threshold),
    the hand-off LLM's answer; `probability`/`inputTokens` `None`. Counts: handed off, and in
    `overLimit`; not a decision-call failure; `handoffCallFailures` does not exist (guard)."""
    _handoff_workspace(tmp_path)
    lms = HandoffStub(decide_responses=list(_OVER_LIMIT_SECOND), chat_responses=[_reply(True)])
    run, _ = _run(tmp_path, lms)
    item = run.items[1]
    assert item.itemId == "fx-cs-02"
    assert item.detail == {
        "path": "turns",
        "tier": "clear_suspend",
        "decidedBy": "handoff",
        "probability": None,
        "inputTokens": None,
        "reason": "over-limit",
    }
    assert [c.wallClockMs for c in item.timing.calls] == [3000.0]
    # the failed decision call (40 ms) plus the hand-off (3000 ms), impl review P9-M1
    assert item.timing.wallClockMs == pytest.approx(_FAILED_DECIDE_MS + 3000.0)
    assert item.timing.unexplainedMs == pytest.approx(2300.0)  # a gap check would withhold it
    assert item.timing.withheldFor is None
    assert (item.outcome, item.counts) == (  # the LLM advanced
        "pass", {"falseAdvanceRateByTurns": 1, "falseAdvanceRate": 1}
    )
    assert len(lms.chat_calls) == 1 and lms.chat_calls[0]["timeout_s"] == 180.0
    assert _counts(_metric(run, "handoffShare")) == (1, 4)
    assert _counts(_metric(run, "overLimit")) == (1, 4)
    assert _counts(_metric(run, "decisionCallFailures")) == (0, 4)
    assert not _has_metric(run, "handoffCallFailures")


@pytest.mark.parametrize(
    ("failure", "withheld", "outcome"),
    [
        (LMStudioCallTimeout("chat timed out"), "timeout", "fail"),
        (LMStudioCallFailed("HTTP 503", status=503, body=""), "no_response", "unrunnable"),
    ],
    ids=["timeout", "failure"],
)
def test_over_limit_hand_off_fails_withhold_column(tmp_path, failure, withheld, outcome):
    """Plan3 §3.3, "Hand-off fails, `withhold` role (guard)", with the architect's ruling: no
    answer, no `failedCall`, `detail = {path, tier, reason}` (the scorer's, unpatched),
    `wallClockMs=None`, `calls=()`, `withheldFor` the failed hand-off call's — even after a
    snapshot without the decision model, since a failing call takes its own and no load producer
    runs. Counts: `overLimit` k and n; `handoffShare` "not attempted" (out of its n)."""
    _handoff_workspace(tmp_path)
    both = _resident("laya", QWEN)
    sequence = [both, both, both, _resident(QWEN), both, both, both, both]
    lms = HandoffStub(
        decide_responses=list(_OVER_LIMIT_SECOND),
        chat_responses=[failure],
        residency_sequence=sequence,
    )
    run, _ = _run(tmp_path, lms)
    item = run.items[1]
    assert item.detail == {"path": "turns", "tier": "clear_suspend", "reason": "over-limit"}
    assert item.timing == ItemTiming(wallClockMs=None, calls=(), withheldFor=withheld)
    assert (item.outcome, item.scoreable, item.counts) == (outcome, {}, {})
    assert _counts(_metric(run, "overLimit")) == (1, 4)
    assert _counts(_metric(run, "handoffShare")) == (0, 3)
    assert _counts(_metric(run, "decisionCallFailures")) == (0, 4)
    assert not _has_metric(run, "handoffCallFailures")
    assert run.latency.latencyWithheldForNoResponse == 1


def test_over_limit_hand_off_fails_fail_closed_column(tmp_path, fail_closed):
    """Plan3 §3.3, "Hand-off fails, `fail-closed` role": kept — `failedCall=handoff`, no
    `decidedBy`, `calls=()` (the decision call failed), the item clock at the raise,
    `withheldFor=None`, scored as the fail-closed answer by the scorer (`answer=None`). Counts:
    not handed off; `overLimit`; `handoffCallFailures`; not a decision-call failure."""
    _handoff_workspace(tmp_path)
    lms = HandoffStub(
        decide_responses=list(_OVER_LIMIT_SECOND), chat_responses=[LMStudioCallTimeout("t")]
    )
    run, _ = _run(tmp_path, lms)
    item = run.items[1]
    assert item.detail == {
        "path": "turns", "tier": "clear_suspend", "failedCall": "handoff", "reason": "over-limit",
    }
    assert item.timing == ItemTiming(
        wallClockMs=pytest.approx(_FAILED_DECIDE_MS + _FAILED_CHAT_MS), calls=(), withheldFor=None
    )
    assert item.outcome == "fail"
    assert fail_closed.handoff_failures == [
        {"answer": None, "outcome": "fail", "decided_by": None, "probability": None,
         "reason": "over-limit", "input_tokens": None}
    ]
    assert _counts(_metric(run, "handoffShare")) == (0, 4)
    assert _counts(_metric(run, "overLimit")) == (1, 4)
    assert _counts(_metric(run, "handoffCallFailures")) == (1, 4)
    assert _counts(_metric(run, "decisionCallFailures")) == (0, 4)


def test_over_limit_hand_off_succeeds_on_a_fail_closed_role_too(tmp_path, fail_closed):
    """The success column is the same on either role: handed off, and `handoffCallFailures`
    0/n is recorded on a `fail-closed` role."""
    _handoff_workspace(tmp_path)
    lms = HandoffStub(decide_responses=list(_OVER_LIMIT_SECOND), chat_responses=[_reply(False)])
    run, _ = _run(tmp_path, lms)
    assert run.items[1].detail["decidedBy"] == "handoff"
    assert _counts(_metric(run, "handoffShare")) == (1, 4)
    assert _counts(_metric(run, "handoffCallFailures")) == (0, 4)
    assert fail_closed.handoff_failures == []


@pytest.mark.parametrize(
    "case", ["another-500-body", "body-document-at-400", "marker-upper-cased-at-500"]
)
def test_a_decision_failure_without_the_marker_is_kept_and_never_handed_off(tmp_path, case):
    """Plan3 §5 U2: a 500 without the marker (and the marker at 400, or upper-cased) stays a
    kept decision failure: no chat call, `failedCall=decision`, no `reason`; it counts in
    `decisionCallFailures`, never in `overLimit` or `handoffShare`'s k."""
    assert base._OVER_LIMIT_ROWS[case]["overLimit"] is False
    _handoff_workspace(tmp_path)
    responses = [base._decided(0.95), base._failure(case), *base._four(0.05, 0.95)]
    lms = HandoffStub(decide_responses=responses)
    run, _ = _run(tmp_path, lms)
    item = run.items[1]
    assert lms.chat_calls == []
    assert item.detail == {
        "path": "turns", "tier": "clear_suspend", "failedCall": "decision",
        "probability": None, "inputTokens": None,
    }
    assert item.timing == ItemTiming(
        wallClockMs=pytest.approx(_FAILED_DECIDE_MS), calls=(), withheldFor=None
    )
    assert _counts(_metric(run, "decisionCallFailures")) == (1, 4)
    assert _counts(_metric(run, "overLimit")) == (0, 4)
    assert _counts(_metric(run, "handoffShare")) == (0, 4)


def test_a_decision_timeout_is_kept_and_never_handed_off(tmp_path):
    _handoff_workspace(tmp_path)
    responses = [LMStudioCallTimeout("decide timed out"), *base._four(0.95, 0.05, 0.95)]
    lms = HandoffStub(decide_responses=responses)
    run, _ = _run(tmp_path, lms)
    assert lms.chat_calls == []
    assert run.items[0].detail["failedCall"] == "decision"
    assert "reason" not in run.items[0].detail
    assert _counts(_metric(run, "decisionCallFailures")) == (1, 4)


# --- a policy hand-off whose chat call fails (plan2 §3.3; plan2 §5 U2) ----------------------------


@pytest.mark.parametrize(
    ("failure", "withheld"),
    [(LMStudioCallTimeout("chat timed out"), "timeout"),
     (LMStudioCallFailed("HTTP 503", status=503, body=""), "no_response")],
    ids=["timeout", "failure"],
)
def test_a_guard_policy_hand_off_failure_is_withheld_exactly_as_today(tmp_path, failure, withheld):
    """Plan2 §5 U2: a guard hand-off timeout is withheld (`wallClockMs=None`) — today's chat-
    surface timing and outcome; out of `handoffShare`'s n, and no `handoffCallFailures` on the
    guard. Architect ruling U2.2 (a): it goes through `score_answer`, and its decision call
    returned, so `detail` keeps that call's `inputTokens` (plan2 §3.4) and nothing else."""
    _handoff_workspace(tmp_path)
    lms = HandoffStub(decide_responses=list(_MIXED), chat_responses=[failure, _reply(False)])
    run, _ = _run(tmp_path, lms)
    item = run.items[2]
    timing = ItemTiming(wallClockMs=None, calls=(), withheldFor=withheld)
    pack = load_pack(tmp_path / "packs" / base._PACK_ID)
    as_today = classification.score_item(_items(tmp_path)[2], None, timing, pack=pack)
    assert (item.outcome, item.scoreable, item.counts) == (
        as_today.outcome, as_today.scoreable, as_today.counts
    )
    assert item.timing == timing
    assert item.detail == {"path": "understanding", "tier": "clear_advance", "inputTokens": 40}
    assert _counts(_metric(run, "handoffShare")) == (1, 3)
    assert _counts(_metric(run, "decisionCallFailures")) == (0, 4)
    assert not _has_metric(run, "handoffCallFailures")


def test_a_fail_closed_policy_hand_off_failure_is_kept_with_only_the_decision_call(
    tmp_path, fail_closed
):
    """Plan2 §5 U2: a join-in-fixture hand-off timeout is kept, with `failedCall="handoff"` and
    `calls` holding only the decision call; the item clock at the raise; not handed off. The
    decision call returned, so its `inputTokens` reaches the scorer (plan2 §3.4)."""
    _handoff_workspace(tmp_path)
    lms = HandoffStub(
        decide_responses=list(_MIXED), chat_responses=[LMStudioCallTimeout("t"), _reply(False)]
    )
    run, _ = _run(tmp_path, lms)
    item = run.items[2]
    assert item.detail == {
        "path": "understanding", "tier": "clear_advance", "failedCall": "handoff"
    }
    assert [c.wallClockMs for c in item.timing.calls] == [12.0]
    # the decision call (12 ms) plus the failed hand-off (25 ms), impl review P9-M1
    assert item.timing.wallClockMs == pytest.approx(12.0 + _FAILED_CHAT_MS)
    assert item.timing.withheldFor is None
    assert fail_closed.handoff_failures == [
        {"answer": None, "outcome": "fail", "decided_by": None, "probability": None,
         "reason": None, "input_tokens": 40}
    ]
    assert _counts(_metric(run, "handoffShare")) == (1, 4)  # item 4 handed off, item 3 kept
    assert _counts(_metric(run, "handoffCallFailures")) == (1, 4)


def test_both_failure_counts_equal_the_injected_failures(tmp_path, fail_closed):
    """Plan2 §5 U2: one decision timeout and two hand-off failures (one over-limit) on a
    `fail-closed` role give `decisionCallFailures` 1/4 and `handoffCallFailures` 2/4 — never one
    counted in the other."""
    _handoff_workspace(tmp_path)
    responses = [LMStudioCallTimeout("d"), base._decided(0.5), _over_limit(), base._decided(0.95)]
    lms = HandoffStub(
        decide_responses=responses,
        chat_responses=[LMStudioCallFailed("x", status=503, body=""), LMStudioCallTimeout("t")],
    )
    run, _ = _run(tmp_path, lms)
    assert [i.detail.get("failedCall") for i in run.items] == [
        "decision", "handoff", "handoff", None
    ]
    assert _counts(_metric(run, "decisionCallFailures")) == (1, 4)
    assert _counts(_metric(run, "handoffCallFailures")) == (2, 4)
    assert _counts(_metric(run, "overLimit")) == (1, 4)
    assert _counts(_metric(run, "handoffShare")) == (0, 4)


def test_a_guard_run_with_a_withheld_policy_hand_off_and_a_decision_failure_counts_apart(tmp_path):
    _handoff_workspace(tmp_path)
    responses = [LMStudioCallTimeout("d"), base._decided(0.5), *base._four(0.95, 0.5)]
    lms = HandoffStub(
        decide_responses=responses, chat_responses=[LMStudioCallTimeout("t"), _reply(True)]
    )
    run, _ = _run(tmp_path, lms)
    assert _counts(_metric(run, "decisionCallFailures")) == (1, 4)
    assert _counts(_metric(run, "handoffShare")) == (1, 3)
    assert _counts(_metric(run, "overLimit")) == (0, 4)


def test_the_counts_are_exploratory_and_never_verdict_metrics(tmp_path):
    _handoff_workspace(tmp_path)
    run, _ = _run(tmp_path, HandoffStub(decide_responses=list(_MIXED)))
    names = {m.name for m in run.aggregates.perClass}
    assert {"handoffShare", "overLimit", "decisionCallFailures"} <= names
    pack = load_pack(tmp_path / "packs" / base._PACK_ID)
    assert not {"handoffShare", "overLimit", "decisionCallFailures"} & set(
        pack.manifest["metrics"]["verdictMetrics"]
    )


# ==================================================================================================
# 4. Plan3 §3.7 on the `handoff` surface: the load step, both warm-ups, then co-residency
# ==================================================================================================


def test_both_models_absent_at_start_are_loaded_then_both_warmed_up(tmp_path):
    """Plan3 §3.7: every model of the arm absent at start gets one `load_model`, before the
    warm-ups; then both warm-ups, the decision model's first; then the instance-config read."""
    _handoff_workspace(tmp_path)
    lms = HandoffStub(decide_responses=list(_MIXED), initially_loaded=())
    run, _ = _run(tmp_path, lms, firstCallTimeoutSeconds=240.0)
    assert lms.load_calls == [
        {"model": "laya", "timeout_s": 240.0}, {"model": QWEN, "timeout_s": 240.0}
    ]
    assert lms.order == [
        ("load_model", "laya"),
        ("load_model", QWEN),
        ("warm_up", "laya"),
        ("warm_up", QWEN),
        ("loaded_instances", "laya"),
    ]
    assert [(c["model"], c["call_surface"]) for c in lms.warm_up_calls] == [
        ("laya", "decisions"), (QWEN, "chat")
    ]
    assert lms.warm_up_calls[1]["timeout_s"] == 240.0
    assert lms.warm_up_calls[1]["system_prompt"] == load_pack(
        tmp_path / "packs" / base._PACK_ID
    ).prompt_config().systemPrompt
    assert lms.was_resident_before == [True, True]
    assert run.fingerprint.get("residentModelsAtStart") == []


def test_a_model_resident_at_start_is_used_as_found_and_only_the_absent_one_loads(tmp_path):
    _handoff_workspace(tmp_path)
    lms = HandoffStub(decide_responses=list(_MIXED), initially_loaded=("laya",))
    _run(tmp_path, lms)
    assert lms.load_calls == [{"model": QWEN, "timeout_s": 300.0}]


def test_a_failed_hand_off_model_load_refuses_with_exit_3_before_any_warm_up(tmp_path):
    _handoff_workspace(tmp_path)
    lms = HandoffStub(
        initially_loaded=("laya",), load_failure=LMStudioCallFailed("x", status=500, body="")
    )
    with pytest.raises(RunRefused) as caught:
        _run(tmp_path, lms)
    assert caught.value.exitCode == 3 and repr(QWEN) in str(caught.value)
    assert lms.warm_up_calls == [] and lms.decide_calls == [] and lms.chat_calls == []


def test_a_decision_model_evicted_by_the_chat_warm_up_refuses_with_exit_3_scoring_nothing(
    tmp_path,
):
    """Plan3 §3.7 / §5 U2: co-residency is checked **after both** warm-ups. A chat warm-up that
    JIT-evicts laya leaves step 7's snapshot without it: exit 3 before any item — no scored
    decide or chat call, no instance-config read. "Nothing stored" is the `cli` test below:
    `run_pack` never stores."""
    _handoff_workspace(tmp_path)
    lms = HandoffStub(decide_responses=list(_MIXED), evict_on_chat_warm_up=True)
    with pytest.raises(RunRefused) as caught:
        _run(tmp_path, lms)
    assert caught.value.exitCode == 3 and "'laya'" in str(caught.value)
    assert "co-residency" in str(caught.value)
    assert lms.order == [("warm_up", "laya"), ("warm_up", QWEN)]
    assert lms.decide_calls == [] and lms.chat_calls == []


def test_cli_run_with_a_decision_model_evicted_by_the_chat_warm_up_exits_3_storing_nothing(
    tmp_path, monkeypatch, capsys
):
    """Impl review P9-n1: only `cli` stores, after `run_pack` returns, so the "stores nothing"
    half of the eviction refusal is asserted through `cli.main` — exit 3 and no `results/`."""
    from modelbench import cli

    _handoff_workspace(tmp_path, contract_valid=True)
    lms = HandoffStub(decide_responses=list(_MIXED), evict_on_chat_warm_up=True)
    monkeypatch.setattr(cli, "LMStudio", lambda base_url: lms)
    code = cli.main(
        ["run", "--root", str(tmp_path), "--pack", base._PACK_ID, "--candidate",
         base._CANDIDATE_ID]
    )
    assert code == 3, capsys.readouterr().err
    assert "co-residency" in capsys.readouterr().err
    assert lms.decide_calls == [] and lms.chat_calls == []
    assert not (tmp_path / "results").exists()


def test_a_hand_off_model_absent_after_the_warm_ups_refuses_with_exit_3(tmp_path):
    _handoff_workspace(tmp_path)
    both = _resident("laya", QWEN)
    lms = HandoffStub(residency_sequence=[both, _resident("laya")])  # start, step 7
    with pytest.raises(RunRefused) as caught:
        _run(tmp_path, lms)
    assert caught.value.exitCode == 3 and repr(QWEN) in str(caught.value)
    assert lms.decide_calls == [] and lms.chat_calls == []


@pytest.mark.parametrize(
    "failure",
    [LMStudioCallTimeout("slow"), LMStudioCallFailed("HTTP 500", status=500, body="")],
    ids=["timeout", "failure"],
)
def test_a_failed_chat_warm_up_refuses_with_exit_3(tmp_path, failure):
    _handoff_workspace(tmp_path)
    lms = HandoffStub(chat_warm_up_failure=failure)
    with pytest.raises(RunRefused) as caught:
        _run(tmp_path, lms)
    assert caught.value.exitCode == 3
    assert lms.decide_calls == [] and lms.chat_calls == []


def test_warmup_extras_warm_each_model_and_the_re_check_compares_with_the_mandatory_one(tmp_path):
    _handoff_workspace(tmp_path)
    lms = HandoffStub(decide_responses=list(_MIXED))
    lms.warm_up_sequence = [base._decided(0.5), base._decided(0.6)]  # mandatory, then the extra
    run, _ = _run(tmp_path, lms, warmupExtra=1)
    assert [(c["model"], c["call_surface"]) for c in lms.warm_up_calls] == [
        ("laya", "decisions"), ("laya", "decisions"), (QWEN, "chat"), (QWEN, "chat")
    ]
    assert run.basis == "by-construction"  # the re-check (0.5) equals the mandatory warm-up


def test_the_instance_config_is_read_under_the_decision_models_id_not_the_chat_models(tmp_path):
    """Carried item (U2.1, impl review P8-m1): `decisionInstanceConfig` is the **decision**
    model's. On this surface the run also has a chat model, whose instance config differs; a
    read under its id would record qwen's config."""
    _handoff_workspace(tmp_path)
    lms = HandoffStub(decide_responses=list(_MIXED))
    run, _ = _run(tmp_path, lms)
    assert [o for o in lms.order if o[0] == "loaded_instances"] == [("loaded_instances", "laya")]
    assert run.fingerprint.get("decisionInstanceConfig") == base._LAYA_CONFIG
    assert run.fingerprint.get("decisionInstanceConfig") != _QWEN_CONFIG


# ==================================================================================================
# 5. The load rule and re-check on the `handoff` surface (plan2 §3.4, §4.1 U2)
# ==================================================================================================


def test_an_equal_re_check_on_the_decision_model_records_by_construction(tmp_path):
    _handoff_workspace(tmp_path)
    lms = HandoffStub(decide_responses=list(_MIXED))
    run, _ = _run(tmp_path, lms)
    recheck = lms.decide_calls[-1]
    assert (recheck["input"], recheck["model"], recheck["timeout_s"]) == (
        WARMUP_DECISION_INPUT, "laya", 2.0
    )
    assert (run.designEffect, run.basis) == (1.0, "by-construction")


def test_a_re_check_with_a_different_probability_records_assumed(tmp_path):
    _handoff_workspace(tmp_path)
    lms = HandoffStub(decide_responses=[*_MIXED, base._decided(0.51)])
    run, _ = _run(tmp_path, lms)
    assert run.basis == "assumed"


def test_a_load_withheld_item_records_assumed_with_no_re_check_call(tmp_path):
    _handoff_workspace(tmp_path)
    both = _resident("laya", QWEN)
    sequence = [both, both, _resident("laya"), both, both, both, both, both]
    lms = HandoffStub(decide_responses=list(_MIXED), residency_sequence=sequence)
    run, _ = _run(tmp_path, lms)
    assert len(lms.decide_calls) == 4 and run.basis == "assumed"


@pytest.mark.parametrize(
    "failure",
    [LMStudioCallTimeout("chat timed out"), LMStudioCallFailed("HTTP 503", status=503, body="")],
    ids=["timeout", "failure"],
)
def test_a_withheld_hand_off_failure_after_a_snapshot_without_laya_still_records_assumed(
    tmp_path, failure
):
    """Impl review P9-m1 / note §Q6 ("one load only if nothing shows a reload"): laya is absent
    before item 3, which hands off and whose chat call fails. The item takes its own
    `withheldFor`, hiding the snapshot, but the run still shows the reload: `assumed`, with no
    re-check call (exactly the 4 scored `decide` calls)."""
    _handoff_workspace(tmp_path)
    both = _resident("laya", QWEN)
    sequence = [both, both, both, both, _resident(QWEN), both, both, both]
    lms = HandoffStub(
        decide_responses=list(_MIXED), chat_responses=[failure], residency_sequence=sequence
    )
    run, _ = _run(tmp_path, lms)
    assert run.items[2].timing.withheldFor in ("timeout", "no_response")
    assert all(i.timing.withheldFor != "load" for i in run.items)
    assert len(lms.decide_calls) == 4 and run.basis == "assumed"


def test_a_fail_closed_policy_hand_off_failure_after_a_snapshot_without_qwen_is_load_withheld(
    tmp_path, fail_closed
):
    """Impl review P9-m2: a kept (`fail-closed`) hand-off failure is `withheldFor=None` "unless
    residency withholds it for load" (plan2 §3.3) — residency of **both** models."""
    _handoff_workspace(tmp_path)
    both = _resident("laya", QWEN)
    sequence = [both, both, both, both, _resident("laya"), both, both, both]
    lms = HandoffStub(
        decide_responses=list(_MIXED), chat_responses=[LMStudioCallTimeout("t")],
        residency_sequence=sequence,
    )
    run, _ = _run(tmp_path, lms)
    item = run.items[2]
    assert item.detail.get("failedCall") == "handoff"
    assert item.timing.withheldFor == "load"


# ==================================================================================================
# 6. The fingerprint, the attestation trip-wire, latency and the stored record
# ==================================================================================================

_HANDOFF_FIELDS = {
    "handoffModelKey": QWEN,
    "handoffModelPublisher": "qwen",
    "handoffArch": "qwen3",
    "handoffQuantization": "Q4_K_M",
    "handoffCompatibilityType": "gguf",
    "handoffMaxContextLength": 262144,
    "handoffModelType": "llm",
    "handoffTemperature": 0.0,
    "handoffMaxTokens": 256,
}


def test_the_handoff_fingerprint_carries_the_hand_off_models_own_identity(tmp_path):
    """v1.3 §3.3 / §4 U2: the `handoff*` fields from qwen's catalog entry, its loaded context
    from the post-load catalog, its runtime from **its** chat warm-up, and the pack's sampling;
    the decision model's identity stays under the bare names."""
    qwen = base._qwen()
    loaded_qwen = type(qwen)(**{**qwen.__dict__, "loaded_context_length": 16384})
    after = [base._laya(loaded=4096), loaded_qwen]
    _handoff_workspace(tmp_path)
    lms = HandoffStub(
        decide_responses=list(_MIXED),
        catalog_after=after,
        chat_runtime={"name": "llama.cpp-cuda", "version": "1.60.1"},
    )
    host = json.loads((tmp_path / "host.json").read_text())
    host["observedAtAttestation"].update(runtimeName="llama.cpp-cuda", runtimeVersion="1.60.1")
    (tmp_path / "host.json").write_text(json.dumps(host))
    run, _ = _run(tmp_path, lms)
    fp = run.fingerprint
    assert fp.armProfile == "model:handoff" and fp.callSurface == "handoff"
    for name, value in _HANDOFF_FIELDS.items():
        assert fp.get(name) == value, name
    assert fp.get("handoffLoadedContextLength") == 16384
    assert (fp.get("handoffRuntimeName"), fp.get("handoffRuntimeVersion")) == (
        "llama.cpp-cuda", "1.60.1"
    )
    assert (fp.get("modelKey"), fp.get("loadedContextLength"), fp.get("modelType")) == (
        "laya", 4096, "decision"
    )
    assert fp.get("candidateId") == base._CANDIDATE_ID
    for bare in ("runtimeName", "runtimeVersion", "temperature", "maxTokens"):
        assert bare not in fp.fields, bare
    assert run.attestationTripWire == "compared"


def test_the_attestation_trip_wire_is_fed_by_the_hand_off_models_chat_warm_up(tmp_path):
    """v1.3 §3.3: the trip-wire compares the hand-off model's chat runtime; a version that moved
    since attestation is stale (exit 5) before any item."""
    _handoff_workspace(tmp_path)
    lms = HandoffStub(chat_runtime={"name": "llama.cpp", "version": "9.9.9"})
    with pytest.raises(RunRefused) as caught:
        _run(tmp_path, lms)
    assert caught.value.exitCode == 5
    assert lms.decide_calls == [] and lms.chat_calls == []


def test_latency_block_treats_handoff_like_embeddings_not_chat(tmp_path):
    _handoff_workspace(tmp_path)
    run, _ = _run(tmp_path, HandoffStub(decide_responses=list(_MIXED)))
    block = latency_block(run.items, call_surface="handoff")
    assert block == run.latency
    assert block.statsCoveredCount is None and block.ttftMsMedian is None
    assert block.unexplainedMsMax is None
    assert block.callCount == 6 and block.latencyTimedCount == 4
    assert latency_block(run.items, call_surface="chat").statsCoveredCount == 2


def test_build_fingerprint_takes_handoff_fields_iff_the_surface_is_handoff():
    from datetime import datetime, timezone

    pack = load_pack(base._FIXTURE_PACK)
    common = dict(
        pack=pack,
        model_info=base._laya(loaded=4096),
        host={"attested": {}},
        started_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
        ended_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
        resident_at_start=[],
        resident_at_end=[],
        residency_source="lmstudio-api-v0",
        runtime_name=None,
        runtime_version=None,
        catalog_after_load=[base._laya(loaded=4096)],
        prompt_config=pack.prompt_config(),
    )
    decision = {"candidateId": "c"}
    with pytest.raises(ValueError, match="handoff_fields are given iff"):
        _build_fingerprint(
            call_surface="handoff", decision_fields=decision, handoff_fields=None, **common
        )
    with pytest.raises(ValueError, match="handoff_fields are given iff"):
        _build_fingerprint(
            call_surface="decisions", decision_fields=decision,
            handoff_fields={"handoffModelKey": QWEN}, **common,
        )
    with pytest.raises(ValueError, match="decision_fields are given iff"):
        _build_fingerprint(
            call_surface="handoff", decision_fields=None,
            handoff_fields={"handoffModelKey": QWEN}, **common,
        )


def test_a_fixture_hand_off_run_stores_a_valid_model_handoff_record_offline(tmp_path):
    """v1.3 §4 U2's done condition: an offline fixture run shows a mix of decided and handed-off
    items with correct timing tuples, stored and read back valid, under `tmp_path`."""
    from modelbench.results import _index_row

    _handoff_workspace(tmp_path)
    lms = HandoffStub(decide_responses=list(_MIXED), chat_responses=[_reply(True), _reply(False)])
    run, disclosures = _run(tmp_path, lms)
    assert disclosures == ()
    path = store(run, tmp_path)
    assert path.parent == tmp_path / "results" / "runs"
    valid, invalid = load_history(tmp_path, packId=base._PACK_ID)
    assert invalid == [] and len(valid) == 1
    stored = valid[0]
    assert stored.fingerprint.armProfile == "model:handoff"
    assert stored.armKey == base._CANDIDATE_ID and stored.modelKey == "laya"
    assert [i.detail["decidedBy"] for i in stored.items] == [
        "decision-model", "decision-model", "handoff", "handoff"
    ]
    assert [len(i.timing.calls) for i in stored.items] == [1, 1, 2, 2]
    assert stored.latency.statsCoveredCount is None
    row = _index_row(stored, valid=True)
    assert (row["candidateId"], row["handoffModelKey"]) == (base._CANDIDATE_ID, QWEN)


# ==================================================================================================
# 7. Per-path diagnostics (v1.3 §3.5, §4 U2; AC-2)
# ==================================================================================================


def test_per_path_diagnostics_split_each_verdict_metric_by_who_decided(tmp_path):
    """The fixture's verdict metrics are the four per-kind rates (architect ruling on impl review
    P12-m2), so each is split by who decided. Item table, one item per cell:
    fx-cs-01 (understanding, label no) decided yes — a false advance; fx-cs-02 (turns, no) decided
    no; fx-ca-01 (understanding, yes) handed off and advanced; fx-ca-02 (turns, yes) handed off and
    suspended — a false suspend. Each cell's `(k, n)` is its one item on that path, or (0, 0)."""
    _handoff_workspace(tmp_path)
    lms = HandoffStub(decide_responses=list(_MIXED), chat_responses=[_reply(True), _reply(False)])
    run, _ = _run(tmp_path, lms)
    expected = {
        "falseAdvanceRateByUnderstandingDecidedByModel": (1, 1),
        "falseAdvanceRateByUnderstandingHandedOff": (0, 0),
        "falseAdvanceRateByTurnsDecidedByModel": (0, 1),
        "falseAdvanceRateByTurnsHandedOff": (0, 0),
        "falseSuspendRateByUnderstandingDecidedByModel": (0, 0),
        "falseSuspendRateByUnderstandingHandedOff": (0, 1),
        "falseSuspendRateByTurnsDecidedByModel": (0, 0),
        "falseSuspendRateByTurnsHandedOff": (1, 1),
    }
    for name, pair in expected.items():
        assert _counts(_metric(run, name)) == pair, name
    assert _counts(_metric(run, "handoffShare")) == (2, 4)
    pack = load_pack(tmp_path / "packs" / base._PACK_ID)
    verdict = set(pack.manifest["metrics"]["verdictMetrics"])
    names = {m.name for m in run.aggregates.perClass}
    split = {n for n in names if n.endswith(("DecidedByModel", "HandedOff"))}
    assert split == set(expected)  # one pair per verdict metric, and no pooled split
    assert verdict <= names and not verdict & split


def test_a_decisions_surface_run_carries_no_hand_off_counts(tmp_path):
    """The `decisions` surface is unchanged: no `handoffShare`, no per-path split."""
    base._workspace(tmp_path)
    run, _ = base._run(tmp_path, base.DecisionStub(decide_responses=base._four(0.9, 0.1, 0.9, 0.9)))
    assert run.fingerprint.armProfile == "model:decisions"
    for name in (
        "handoffShare", "handoffCallFailures", "falseAdvanceRateByUnderstandingHandedOff"
    ):
        assert not _has_metric(run, name), name


def test_a_kept_decision_failure_after_a_snapshot_missing_the_hand_off_model_is_load_withheld(
    tmp_path,
):
    """Plan2 §3.3: a kept item is `withheldFor=None` "unless residency withholds it for load" —
    on this surface, residency of **both** models."""
    _handoff_workspace(tmp_path)
    both = _resident("laya", QWEN)
    sequence = [both, both, _resident("laya"), both, both, both, both, both]
    responses = [LMStudioCallTimeout("d"), *base._four(0.95, 0.05, 0.95)]
    run, _ = _run(tmp_path, HandoffStub(decide_responses=responses, residency_sequence=sequence))
    item = run.items[0]
    assert item.detail["failedCall"] == "decision"
    assert (item.timing.calls, item.timing.withheldFor) == ((), "load")
