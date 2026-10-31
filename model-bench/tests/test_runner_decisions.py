"""typed-decision-models U1d.2 — the plain `decisions` arm (plan v1.3 §3.3, §4 U1d; plan2 §3.2,
§3.4; plan3 §3.3).

Offline throughout: a hand-built `LMStudio`-shaped stub, an on-disk fixture pack
(`tests/fixtures/packs/decision_point_guard/`) and a candidate written under `tmp_path`. Nothing
here writes under `model-bench/results/` (plan2 §4.2 / plan3 §4.1: no `model:decisions` record is
stored there before U1e).

Sections, in the unit's landing order: the scorer seam's three hooks; the packs half; the runner
(refusals, the per-item flow, `latency_block`, the fingerprint); the done-condition run.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from modelbench.runner import ItemScorer
from modelbench.scoring import classification

_REAL_GUARD_PACK = Path(__file__).resolve().parents[1] / "packs" / "guard-judge-understanding"

# ==================================================================================================
# 1. The scorer seam: three optional hooks (plan v1.3 §3.3 "Scorer seam")
# ==================================================================================================

#: Transcribed by hand from the plans, never read off either signature: `score_answer`'s
#: keyword-only parameters (v1.3 §3.3 `pack, decided_by, probability`; plan2 §3.3 `failed_call`;
#: plan3 §3.3 `reason`; plan2 §3.4 / the U1d.1 build `input_tokens`). None has a default.
_SCORE_ANSWER_KEYWORDS = (
    "pack", "decided_by", "probability", "failed_call", "reason", "input_tokens",
)


def _shape(func) -> list[tuple[str, inspect._ParameterKind, bool]]:
    """`(name, kind, has_default)` per parameter, `self` dropped — what a structural match needs."""
    params = [p for p in inspect.signature(func).parameters.values() if p.name != "self"]
    return [(p.name, p.kind, p.default is not inspect.Parameter.empty) for p in params]


def test_the_protocol_score_answer_hook_takes_the_plans_keywords_with_no_default():
    keyword_only = [
        (name, has_default)
        for name, kind, has_default in _shape(ItemScorer.score_answer)
        if kind is inspect.Parameter.KEYWORD_ONLY
    ]
    assert keyword_only == [(name, False) for name in _SCORE_ANSWER_KEYWORDS]


def test_classification_satisfies_each_of_the_three_decision_hooks_structurally():
    """U3a's (and U1d.1's) `classification` functions are the hooks' first implementation (v1.3
    §4 U1d): same names, kinds, order and default-ness as the Protocol, hook by hook."""
    for hook in ("decision_input", "answer_from_chat", "score_answer"):
        assert _shape(getattr(classification, hook)) == _shape(getattr(ItemScorer, hook)), hook


def test_guard_decision_input_renders_normalized_turns_by_hand():
    """v1.3 §3.2: `_render_judge_user(condition, understanding, normalized_turns)` — the speaker is
    `displayName` (normalization ran), the empty row is dropped first. Expected text by hand."""
    item = {
        "condition": "the user named a service",
        "understanding": {},
        "turns": [
            {"displayName": "Alice", "authorId": "u1", "role": "user", "text": "checkout is down"},
            {"authorId": "u2", "text": ""},
            {"authorId": "u2", "text": "since 09:20"},
        ],
    }
    expected = (
        "CONDITION: the user named a service\n\n"
        "RECENT TURNS (context only):\nAlice: checkout is down\nu2: since 09:20"
    )
    assert classification.decision_input(item, pack=None) == expected


def test_guard_decision_input_equals_the_chat_judges_user_message_on_every_real_item():
    """"Exactly as today" (v1.3 §3.2): the decision model sees what the chat judge sees."""
    from modelbench.packs import load_pack

    pack = load_pack(_REAL_GUARD_PACK)
    items = list(pack.iter_items())
    assert len(items) == 85
    for item in items:
        user = classification.build_messages(item, pack=pack)[1]["content"]
        assert classification.decision_input(item, pack=pack) == user, item["itemId"]


# ==================================================================================================
# 2. packs.py: the `lmstudio-decisions` token and the answer-space check (plan v1.3 §3.2, §3.3)
# ==================================================================================================

_FIXTURE_PACK = Path(__file__).resolve().parent / "fixtures" / "packs" / "decision_point_guard"


def test_derive_decision_support_reads_the_token_from_a_list_only():
    from modelbench.packs import derive_decision_support

    assert derive_decision_support(["lmstudio-chat", "lmstudio-decisions"]) is True
    assert derive_decision_support(("lmstudio-embeddings", "lmstudio-decisions")) is True
    assert derive_decision_support(["lmstudio-chat"]) is False
    assert derive_decision_support(["lmstudio-decision"]) is False
    assert derive_decision_support("lmstudio-decisions") is False  # a bare string is no list
    assert derive_decision_support(None) is False


def test_the_decisions_token_is_additive_to_the_call_surface_never_one_itself():
    """v1.3 §3.3: exactly one of chat/embeddings, optionally plus `lmstudio-decisions`."""
    from modelbench.packs import derive_call_surface

    assert derive_call_surface(["lmstudio-chat", "lmstudio-decisions"]) == "chat"
    assert derive_call_surface(["lmstudio-embeddings", "lmstudio-decisions"]) == "embeddings"
    assert derive_call_surface(["lmstudio-decisions"]) is None


def test_the_fixture_decision_point_pack_validates_and_declares_both_surfaces():
    from modelbench.packs import derive_decision_support, load_pack, validate_pack

    pack = load_pack(_FIXTURE_PACK)
    assert validate_pack(pack) == []
    assert derive_decision_support(pack.manifest["environment"]["requires"]) is True


def _predicate_question() -> dict:
    return {"type": "predicate", "name": "advance", "instructions": "Does it hold?"}


def _choice_question(*values: str) -> dict:
    return {
        "type": "choice",
        "name": "type",
        "instructions": "Which type?",
        "choices": [{"value": v, "instructions": ""} for v in values],
    }


_CHOICE_POINT = {"answer": {"type": "choice", "labels": ["order", "question", "other"]}}


def test_answer_space_accepts_a_matching_predicate_and_a_choice_in_any_order():
    from modelbench.packs import answer_space_problem

    assert answer_space_problem({"answer": {"type": "predicate"}}, _predicate_question()) is None
    question = _choice_question("other", "order", "question")
    assert answer_space_problem(_CHOICE_POINT, question) is None


def test_answer_space_refuses_a_choice_whose_values_are_not_exactly_the_labels():
    from modelbench.packs import answer_space_problem

    for values in (("order", "question"), ("order", "question", "other", "complaint")):
        problem = answer_space_problem(_CHOICE_POINT, _choice_question(*values))
        assert problem is not None and "labels" in problem, values


def test_answer_space_refuses_a_type_mismatch_both_ways():
    from modelbench.packs import answer_space_problem

    point = {"answer": {"type": "predicate"}}
    assert "'choice' question" in answer_space_problem(point, _choice_question("a", "b"))
    assert "'predicate' question" in answer_space_problem(_CHOICE_POINT, _predicate_question())


def test_answer_space_refuses_a_score_question_up_front():
    from modelbench.packs import answer_space_problem

    question = {"type": "score", "name": "s", "instructions": ""}
    problem = answer_space_problem({"answer": {"type": "predicate"}}, question)
    assert problem is not None and "no answer space accepts" in problem


def test_answer_space_refuses_a_missing_or_malformed_decision_point_block():
    from modelbench.packs import answer_space_problem

    assert "no decisionPoint" in answer_space_problem(None, _predicate_question())
    for block in ({}, {"answer": "predicate"}, {"answer": {"type": "score"}}):
        assert "decisionPoint.answer" in answer_space_problem(block, _predicate_question())
    no_labels = {"answer": {"type": "choice"}}
    assert "labels" in answer_space_problem(no_labels, _choice_question("a"))


# ==================================================================================================
# 3. runner.py: the `decisions` surface (plan v1.3 §3.3; plan2 §3.2, §3.4)
# ==================================================================================================

import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
from dataclasses import replace  # noqa: E402
from types import MappingProxyType  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402

from modelbench import hostinfo  # noqa: E402
from modelbench.decisions import Candidate, eval_set_hash  # noqa: E402
from modelbench.lmstudio import (  # noqa: E402
    DecisionAnswer,
    DecisionResult,
    LMStudioCallFailed,
    LMStudioCallTimeout,
    LoadResult,
    ModelInfo,
    ResidentModel,
)
from modelbench.packs import load_pack  # noqa: E402
from modelbench.results import CallTiming, ItemResult, ItemTiming, load_history, store  # noqa: E402
from modelbench.runner import (  # noqa: E402
    RunConfig,
    RunRefused,
    _drive_decision_items,
    latency_block,
    run_pack,
)

_PACK_ID = "decision-point-guard-fixture"
_CANDIDATE_ID = "guard-fixture-decisions-v1"
_QUESTION = {
    "type": "predicate",
    "name": "advance",
    "instructions": "Has the user named the failing service?",
}
#: The question's canonical JSON, written by hand (keys sorted, no whitespace) — never produced by
#: `question_spec_hash`, so the fingerprint's `questionSpecHash` is pinned from outside.
_QUESTION_CANONICAL = (
    '{"instructions":"Has the user named the failing service?","name":"advance",'
    '"type":"predicate"}'
)
#: The fixture items, in file order, and their gold label (`items.jsonl`).
_ITEM_IDS = ("fx-cs-01", "fx-cs-02", "fx-ca-01", "fx-ca-02")


def _candidate_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "candidateId": _CANDIDATE_ID,
        "role": "guard-judge",
        "decisionModel": "laya",
        "question": dict(_QUESTION),
        # An empty band at 0.8: no hand-off, so no hand-off model is needed (plan3 §3.2).
        "policy": {"kind": "predicate-band", "yesAtOrAbove": 0.8, "noBelow": 0.8},
        "handoffModel": None,
        "decisionTimeoutSeconds": 2.0,
        "fit": {"outcome": "within-margin", "qualityTarget": "D_c<=0"},
        "fitManifest": {
            "file": f"{_CANDIDATE_ID}.fit.txt",
            "sha256": "3" * 64,
            "size": 120,
            "purposes": ["question-tuning", "cutoff"],
        },
        "provenance": {"builtBy": "decision-models", "datasetVersion": "fx-1", "sourceSha": "abc"},
    }
    body.update(overrides)
    return body


def _laya(*, model_type: str = "decision", loaded: int | None = None) -> ModelInfo:
    return ModelInfo(
        id="laya",
        object="model",
        type=model_type,
        publisher="ggml-org",
        arch="modern-bert",
        compatibility_type="gguf",
        quantization="Q8_0",
        state="loaded" if loaded else "not-loaded",
        max_context_length=8192,
        capabilities=None,
        loaded_context_length=loaded,
    )


def _qwen() -> ModelInfo:
    return ModelInfo(
        id="qwen/qwen3-4b-2507",
        object="model",
        type="llm",
        publisher="qwen",
        arch="qwen3",
        compatibility_type="gguf",
        quantization="Q4_K_M",
        state="loaded",
        max_context_length=262144,
        capabilities=("tool_use",),
        loaded_context_length=8192,
    )


_LAYA_RESIDENT = [ResidentModel(id="laya", state="loaded")]


def _decided(p: float, *, usage: Any = None, wall_clock_ms: float = 12.0) -> DecisionResult:
    return DecisionResult(
        answers=(
            DecisionAnswer(
                name="advance",
                type="predicate",
                probability=p,
                choice=None,
                probabilities=None,
                confidence=None,
            ),
        ),
        model="laya",
        usage={"input_tokens": 40, "output_tokens": 0, "total_tokens": 40} if usage is None
        else usage,
        wallClockMs=wall_clock_ms,
    )


class DecisionStub:
    """An `LMStudio`-shaped stub for the `decisions` surface: `decide` pops results in call order
    (an exception instance is raised), `residency` pops snapshots (default: laya resident),
    `catalog` returns the pre-load list once, then the post-load list."""

    def __init__(
        self,
        *,
        decide_responses: list[Any] | None = None,
        residency_sequence: list[list[ResidentModel]] | None = None,
        catalog_before: list[ModelInfo] | None = None,
        catalog_after: list[ModelInfo] | None = None,
    ) -> None:
        self._decide = list(decide_responses or [])
        self._residency = list(residency_sequence or [])
        self._catalogs = [
            catalog_before if catalog_before is not None else [_laya(), _qwen()],
            catalog_after if catalog_after is not None else [_laya(loaded=4096), _qwen()],
        ]
        self.decide_calls: list[dict[str, Any]] = []
        self.warm_up_calls: list[dict[str, Any]] = []
        self.chat_calls: list[Any] = []

    def probe(self) -> str:
        return "api-v0"

    def residency(self) -> list[ResidentModel]:
        return self._residency.pop(0) if self._residency else list(_LAYA_RESIDENT)

    def catalog(self) -> list[ModelInfo]:
        return list(self._catalogs.pop(0) if len(self._catalogs) > 1 else self._catalogs[0])

    def warm_up(
        self, model_key, *, call_surface, system_prompt, was_resident_before, timeout_s,
        question=None,
    ):
        self.warm_up_calls.append(
            {"model": model_key, "call_surface": call_surface, "question": question,
             "timeout_s": timeout_s, "system_prompt": system_prompt}
        )
        return LoadResult(
            wallClockMs=2500.0, wasResidentBefore=was_resident_before, runtime=None, stats=None
        )

    def decide(self, input, questions, *, model, timeout_s):
        self.decide_calls.append(
            {"input": input, "questions": questions, "model": model, "timeout_s": timeout_s}
        )
        response = self._decide.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def chat(self, *args, **kwargs):  # pragma: no cover - the decisions surface never chats
        self.chat_calls.append((args, kwargs))
        raise AssertionError("the decisions surface made a chat call")


def _host_json(root: Path) -> None:
    hostinfo.write_host_info(
        root,
        {
            "schemaVersion": 1,
            "apiBaseUrl": "http://localhost:1234",
            "attested": {
                "lmStudioAppVersion": "0.3.31",
                "kvCacheSetting": "f16",
                "hostRamGb": 16,
                "otherResidentWorkloads": [],
            },
            "attestedAt": "2026-09-10T00:00:00Z",
            "observedAtAttestation": {
                "residencySource": "lmstudio-api-v0",
                "runtimeName": "llama.cpp",
                "runtimeVersion": "1.52.0",
                "runtimeObservedAt": "2026-09-10T00:00:00Z",
            },
        },
    )


def _workspace(root: Path, *, manifest_edit=None, items_edit=None, **candidate) -> Path:
    """`root` laid out as a model-bench root: the fixture pack under `packs/`, `host.json`, and
    the candidate under `candidates/`. Never the shipped `model-bench/` root."""
    pack_dir = root / "packs" / _PACK_ID
    shutil.copytree(_FIXTURE_PACK, pack_dir)
    if manifest_edit is not None:
        manifest = json.loads((pack_dir / "pack.json").read_text())
        manifest_edit(manifest)
        (pack_dir / "pack.json").write_text(json.dumps(manifest))
    if items_edit is not None:
        rows = [json.loads(line) for line in (pack_dir / "items.jsonl").read_text().splitlines()]
        rows = [items_edit(row) for row in rows]
        (pack_dir / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    _host_json(root)
    (root / "candidates").mkdir()
    (root / "candidates" / f"{_CANDIDATE_ID}.json").write_text(
        json.dumps(_candidate_body(**candidate), indent=2)
    )
    return pack_dir


def _cfg(**overrides: Any) -> RunConfig:
    base: dict[str, Any] = dict(
        modelKey=None, sessionId="s-u1d2", referenceKey=None, candidateId=_CANDIDATE_ID
    )
    base.update(overrides)
    return RunConfig(**base)


class _Clock:
    """A monotonic clock in seconds that advances `step` per read: each item reads it twice."""

    def __init__(self, step: float) -> None:
        self.t, self.step = 100.0, step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


def _run(root: Path, lms: DecisionStub, *, clock=None, **cfg: Any):
    pack = load_pack(root / "packs" / _PACK_ID)
    return run_pack(
        pack, _cfg(**cfg), lmstudio=lms, root=root, clock=clock or _Clock(0.25)
    )


def _four(*ps: float, **kw: Any) -> list[DecisionResult]:
    return [_decided(p, **kw) for p in ps]


# --- RunConfig -----------------------------------------------------------------------------------


def test_run_config_takes_exactly_one_of_a_model_and_a_candidate():
    with pytest.raises(ValueError, match="exactly one"):
        RunConfig(modelKey="laya", sessionId=None, referenceKey=None, candidateId="c")
    with pytest.raises(ValueError, match="exactly one"):
        RunConfig(modelKey=None, sessionId=None, referenceKey=None)
    candidate_cfg = RunConfig(modelKey=None, sessionId=None, referenceKey=None, candidateId="c")
    assert candidate_cfg.armKey == "c"
    assert RunConfig(modelKey="m", sessionId=None, referenceKey=None).armKey == "m"


# --- §3.3's refusals, all before any call (exit 4) ------------------------------------------------


def _refused(root: Path, lms: DecisionStub) -> RunRefused:
    with pytest.raises(RunRefused) as caught:
        _run(root, lms)
    assert lms.warm_up_calls == [] and lms.decide_calls == [], "refused after a model call"
    return caught.value


def test_a_candidate_whose_model_has_catalog_type_llm_refuses(tmp_path):
    """v1.3 §5 U1d's first row: catalog `type != "decision"`."""
    _workspace(tmp_path)
    exc = _refused(tmp_path, DecisionStub(catalog_before=[_laya(model_type="llm")]))
    assert exc.exitCode == 4 and "catalog type 'llm'" in str(exc)


def test_a_pack_with_no_decision_point_block_refuses_a_candidate(tmp_path):
    _workspace(tmp_path, manifest_edit=lambda m: m.pop("decisionPoint"))
    exc = _refused(tmp_path, DecisionStub())
    assert exc.exitCode == 4 and "no decisionPoint block" in str(exc)


def test_a_pack_without_the_decisions_token_refuses_a_candidate(tmp_path):
    def chat_only(manifest):
        manifest["environment"]["requires"] = ["lmstudio-chat"]

    _workspace(tmp_path, manifest_edit=chat_only)
    exc = _refused(tmp_path, DecisionStub())
    assert exc.exitCode == 4 and "lmstudio-decisions" in str(exc)


def test_a_candidate_with_a_hand_off_model_refuses_until_the_handoff_surface_exists(tmp_path):
    _workspace(
        tmp_path,
        handoffModel="qwen/qwen3-4b-2507",
        policy={"kind": "predicate-band", "yesAtOrAbove": 0.9, "noBelow": 0.1},
    )
    exc = _refused(tmp_path, DecisionStub())
    assert exc.exitCode == 4 and "handoff surface" in str(exc)


def test_a_candidate_outside_the_packs_answer_space_refuses(tmp_path):
    question = {
        "type": "choice",
        "name": "advance",
        "instructions": "",
        "choices": [{"value": "yes", "instructions": ""}, {"value": "no", "instructions": ""}],
    }
    _workspace(
        tmp_path, question=question, policy={"kind": "choice-confidence", "minConfidence": 0.0}
    )
    exc = _refused(tmp_path, DecisionStub())
    assert exc.exitCode == 4 and "answer space is 'predicate'" in str(exc)


def test_a_candidate_for_another_role_refuses(tmp_path):
    def other_role(manifest):
        manifest["role"] = "nlq-generator"

    _workspace(tmp_path, manifest_edit=other_role)
    exc = _refused(tmp_path, DecisionStub())
    assert exc.exitCode == 4 and "a 'guard-judge' candidate" in str(exc)


def test_an_unloadable_candidate_refuses(tmp_path):
    _workspace(tmp_path, fit={"outcome": "promising", "qualityTarget": "D_c<=0"})
    exc = _refused(tmp_path, DecisionStub())
    assert exc.exitCode == 4 and "fit.outcome" in str(exc)


def test_a_pack_item_with_no_stratum_refuses_rather_than_defaulting_the_eval_set(tmp_path):
    def drop_stratum(row):
        row.pop("stratum")
        return row

    _workspace(tmp_path, items_edit=drop_stratum)
    exc = _refused(tmp_path, DecisionStub())
    assert exc.exitCode == 4 and "evalSetHash" in str(exc)


def test_a_decision_model_absent_from_the_catalog_refuses(tmp_path):
    _workspace(tmp_path)
    exc = _refused(tmp_path, DecisionStub(catalog_before=[_qwen()]))
    assert exc.exitCode == 4 and "'laya'" in str(exc)


def test_a_scorer_without_the_decision_hooks_refuses(tmp_path, monkeypatch):
    class ChatOnlyScorer:
        def score_item(self, *a, **k): ...
        def aggregate(self, *a, **k): ...

    monkeypatch.setattr("modelbench.runner._load_item_scorer", lambda pack: ChatOnlyScorer())
    _workspace(tmp_path)
    exc = _refused(tmp_path, DecisionStub())
    assert exc.exitCode == 4 and "decision_input, score_answer" in str(exc)


# --- the per-item flow ---------------------------------------------------------------------------


def test_the_policy_decides_each_item_not_a_raw_half_threshold(tmp_path):
    """Band at 0.8: p=0.95 yes, 0.7 no, 0.85 yes, 0.1 no. A raw `p >= 0.5` would answer the
    second item yes — a false advance this run must not record."""
    _workspace(tmp_path)
    run, _ = _run(tmp_path, DecisionStub(decide_responses=_four(0.95, 0.7, 0.85, 0.1)))
    counts = {i.itemId: i.counts for i in run.items}
    assert counts == {
        "fx-cs-01": {"falseAdvanceRate": 1},
        "fx-cs-02": {"falseAdvanceRate": 0},
        "fx-ca-01": {"falseSuspendRate": 0},
        "fx-ca-02": {"falseSuspendRate": 1},
    }


def test_every_item_records_decided_by_its_probability_and_its_input_tokens(tmp_path):
    usages = [
        {"input_tokens": 41, "output_tokens": 0, "total_tokens": 41},
        {"input_tokens": 52, "output_tokens": 3, "total_tokens": 99, "prompt_tokens": 7},
        None,  # no usage object at all
        {"input_tokens": True, "output_tokens": 0, "total_tokens": 1},  # a bool is no count
    ]
    responses = [
        DecisionResult(answers=r.answers, model="laya", usage=u, wallClockMs=12.0)
        for r, u in zip(_four(0.95, 0.7, 0.85, 0.1), usages)
    ]
    _workspace(tmp_path)
    run, _ = _run(tmp_path, DecisionStub(decide_responses=responses))
    details = [
        (i.detail["decidedBy"], i.detail["probability"], i.detail["inputTokens"]) for i in run.items
    ]
    assert details == [
        ("decision-model", 0.95, 41),
        ("decision-model", 0.7, 52),
        ("decision-model", 0.85, None),
        ("decision-model", 0.1, None),
    ]
    assert all("failedCall" not in i.detail and "reason" not in i.detail for i in run.items)


def test_each_decide_call_carries_the_rendered_input_question_model_and_decision_timeout(tmp_path):
    from modelbench.scoring import classification

    pack_dir = _workspace(tmp_path, decisionTimeoutSeconds=1.5)
    lms = DecisionStub(decide_responses=_four(0.9, 0.1, 0.9, 0.9))
    _run(tmp_path, lms, requestTimeoutSeconds=120.0, firstCallTimeoutSeconds=300.0)
    pack = load_pack(pack_dir)
    expected_inputs = [classification.decision_input(i, pack=pack) for i in pack.iter_items()]
    assert [c["input"] for c in lms.decide_calls] == expected_inputs
    assert all(c["questions"] == [_QUESTION] for c in lms.decide_calls)
    assert {c["model"] for c in lms.decide_calls} == {"laya"}
    assert {c["timeout_s"] for c in lms.decide_calls} == {1.5}  # plan2 §3.2, not --request-timeout
    assert lms.warm_up_calls == [
        {"model": "laya", "call_surface": "decisions", "question": _QUESTION,
         "timeout_s": 300.0, "system_prompt": None}
    ]


def test_an_item_is_timed_by_the_item_clock_with_the_decision_call_as_its_one_call(tmp_path):
    """Item clock 250 ms (two reads 0.25 s apart); the call's own wall clock 12 ms."""
    _workspace(tmp_path)
    run, _ = _run(tmp_path, DecisionStub(decide_responses=_four(0.9, 0.1, 0.9, 0.9)))
    for item in run.items:
        assert item.timing.wallClockMs == pytest.approx(250.0)
        assert item.timing.calls == (
            CallTiming(
                wallClockMs=12.0, ttftMs=None, generationMs=None, promptTokens=None,
                tokensPerSecond=None,
            ),
        )


def test_an_item_after_a_snapshot_without_the_decision_model_is_withheld_for_load(tmp_path):
    _workspace(tmp_path)
    residency = [
        [],                  # resident at start
        list(_LAYA_RESIDENT),  # baseline (step 7)
        list(_LAYA_RESIDENT),  # before item 1
        [],                  # before item 2: laya gone
        list(_LAYA_RESIDENT),
        list(_LAYA_RESIDENT),
    ]
    lms = DecisionStub(decide_responses=_four(0.9, 0.1, 0.9, 0.9), residency_sequence=residency)
    run, _ = _run(tmp_path, lms)
    assert [i.timing.withheldFor for i in run.items] == [None, "load", None, None]
    assert run.latency.latencyWithheldForLoad == 1


def test_a_failed_decision_call_refuses_the_run_until_u1d3_keeps_it(tmp_path):
    """U1d.2's choice: refuse loudly (exit 3), never a call-failure `withheldFor` through
    `score_answer`. U1d.3 replaces this with the kept item (plan2 §3.3)."""
    for failure, marker in (
        (LMStudioCallTimeout("timed out"), "timed out"),
        (LMStudioCallFailed("HTTP 500", status=500, body="too large to process"), "over-limit"),
        (LMStudioCallFailed("HTTP 500", status=500, body="boom"), "boom"),
    ):
        root = tmp_path / marker.replace(" ", "-")
        root.mkdir()
        _workspace(root)
        with pytest.raises(RunRefused) as caught:
            _run(root, DecisionStub(decide_responses=[_decided(0.9), failure]))
        assert caught.value.exitCode == 3
        assert "U1d.3" in str(caught.value) and "'fx-cs-02'" in str(caught.value)
        assert ("over-limit" in str(caught.value)) == (marker == "over-limit")


def test_a_policy_hand_off_on_the_decisions_surface_refuses_never_scores_as_decided(tmp_path):
    """Unreachable from `Candidate.load` (plan3 §3.2's one-way rule), so the candidate is built
    past it: a non-empty band with no hand-off model. A hand-off-band item must not be recorded
    as decided by the decision model."""
    _workspace(tmp_path)
    loaded = Candidate.load(tmp_path, _CANDIDATE_ID)
    banded = replace(
        loaded,
        policy=MappingProxyType({"kind": "predicate-band", "yesAtOrAbove": 0.9, "noBelow": 0.1}),
    )
    pack = load_pack(tmp_path / "packs" / _PACK_ID)
    from modelbench.scoring import classification

    with pytest.raises(RunRefused) as caught:
        _drive_decision_items(
            pack,
            candidate=banded,
            scorer=classification,
            lmstudio=DecisionStub(decide_responses=[_decided(0.95), _decided(0.5)]),
            model_info=_laya(loaded=4096),
            baseline_residency=list(_LAYA_RESIDENT),
            clock=_Clock(0.01),
        )
    assert caught.value.exitCode == 4 and "'fx-cs-02'" in str(caught.value)


# --- latency_block and the attestation trip-wire on the decisions surface ------------------------


def _decision_item(item_id: str, ms: float) -> ItemResult:
    call = CallTiming(
        wallClockMs=ms - 1.0, ttftMs=None, generationMs=None, promptTokens=None,
        tokensPerSecond=None,
    )
    return ItemResult(
        itemId=item_id,
        pairingKey=(item_id,),
        outcome="pass",
        scoreable={},
        counts={},
        timing=ItemTiming(wallClockMs=ms, calls=(call,), withheldFor=None),
    )


def test_latency_block_treats_decisions_like_embeddings_not_chat():
    """v1.3 §3.3: no `stats` on this surface, so every stats-derived field is `None` — never the
    chat surface's real `0` — and the wall-clock figures apply unchanged."""
    items = [_decision_item(f"i{n}", 20.0 + n) for n in range(4)]
    block = latency_block(items, call_surface="decisions")
    embeddings = latency_block(items, call_surface="embeddings")
    assert block == embeddings
    assert block.statsCoveredCount is None
    assert (block.ttftMsMedian, block.prefillMsPer1kMedian, block.tokensPerSecondMedian) == (
        None, None, None
    )
    assert block.latencyMsP50 == 21.0 and block.callCount == 4
    assert latency_block(items, call_surface="chat").statsCoveredCount == 0


def test_the_attestation_trip_wire_is_unavailable_on_the_decisions_surface(tmp_path):
    """v1.3 §3.3: "For `decisions` alone it is `unavailable`, as for embeddings" — even on a host
    that has never observed a runtime, where a chat run would back-fill one."""
    _host_json(tmp_path)
    host = hostinfo.read_host_info(tmp_path)
    del host["observedAtAttestation"]["runtimeName"]
    check = hostinfo.check_attestation_staleness(
        host,
        call_surface="decisions",
        residency_source="lmstudio-api-v0",
        runtime_name=None,
        runtime_version=None,
    )
    assert (check.outcome, check.stale, check.updated_host) == ("unavailable", False, None)


# --- the fingerprint ------------------------------------------------------------------------------


def test_the_decisions_fingerprint_carries_the_candidate_and_the_catalogs_model_identity(tmp_path):
    pack_dir = _workspace(tmp_path)
    run, _ = _run(tmp_path, DecisionStub(decide_responses=_four(0.9, 0.1, 0.9, 0.9)))
    fp = run.fingerprint
    assert fp.armProfile == "model:decisions"
    candidate_bytes = (tmp_path / "candidates" / f"{_CANDIDATE_ID}.json").read_bytes()
    pack = load_pack(pack_dir)
    expected = {
        "candidateId": _CANDIDATE_ID,
        "candidateContentHash": hashlib.sha256(candidate_bytes).hexdigest(),
        "questionSpecHash": hashlib.sha256(_QUESTION_CANONICAL.encode()).hexdigest(),
        "decisionPolicy": {"kind": "predicate-band", "yesAtOrAbove": 0.8, "noBelow": 0.8},
        "fitManifestHash": "3" * 64,
        "fitManifestSize": 120,
        "evalSetHash": eval_set_hash(pack.iter_items(), "guard-judge"),
        "fitOutcome": "within-margin",
        "fitQualityTarget": "D_c<=0",
        # model identity: the catalog's decision-model entry, never the pack's incumbent
        "modelKey": "laya",
        "modelPublisher": "ggml-org",
        "arch": "modern-bert",
        "quantization": "Q8_0",
        "modelType": "decision",
        "modelCapabilities": [],
        "modelCapabilitiesPresent": False,
        "loadedContextLength": 4096,  # from the catalog read after the load
    }
    assert {k: fp.get(k) for k in expected} == expected
    for forbidden in ("runtimeName", "runtimeVersion", "temperature", "maxTokens"):
        assert fp.get(forbidden, "ABSENT") == "ABSENT", forbidden
    assert fp.validate() == []


# ==================================================================================================
# 4. The done-condition (v1.3 §4 U1d): a fixture pack plus a fixture candidate produces a valid
#    stored `model:decisions` run, offline, against a fake LM Studio — stored under tmp_path only.
# ==================================================================================================


def test_a_fixture_candidate_run_stores_a_valid_model_decisions_record_offline(tmp_path):
    _workspace(tmp_path)
    run, disclosures = _run(
        tmp_path, DecisionStub(decide_responses=_four(0.95, 0.1, 0.85, 0.9))
    )
    assert disclosures == ()
    assert run.runId.startswith(f"{_PACK_ID}-{_CANDIDATE_ID}-")
    assert run.armKind == "model" and run.attestationTripWire == "unavailable"
    assert (run.designEffect, run.basis) == (1.0, "by-construction")
    assert run.aggregates.n == 4

    path = store(run, tmp_path)
    assert path.parent == tmp_path / "results" / "runs"
    valid, invalid = load_history(tmp_path, packId=_PACK_ID)
    assert invalid == [] and len(valid) == 1
    stored = valid[0]
    assert stored.fingerprint.armProfile == "model:decisions"
    assert stored.armKey == _CANDIDATE_ID and stored.modelKey == "laya"
    assert stored.latency.statsCoveredCount is None
    assert stored.latency.latencyItemCount == 4
    assert [i.detail["decidedBy"] for i in stored.items] == ["decision-model"] * 4
    assert tuple(i.itemId for i in stored.items) == _ITEM_IDS


# ==================================================================================================
# 5. cli.py and results.py: `run --candidate`, `RunResult.armKey`, the `index.csv` columns
# ==================================================================================================

from modelbench import cli  # noqa: E402
from modelbench.results import _index_row, rebuild_index  # noqa: E402


def test_run_refuses_candidate_and_model_together_and_neither(tmp_path, capsys):
    _workspace(tmp_path)
    base = ["run", "--root", str(tmp_path), "--pack", _PACK_ID]
    assert cli.main([*base, "--candidate", _CANDIDATE_ID, "--model", "laya"]) == cli.EXIT_USAGE
    assert "not allowed with" in capsys.readouterr().err
    assert cli.main(base) == cli.EXIT_USAGE


def test_run_candidate_builds_a_candidate_config_with_no_model_key(tmp_path, monkeypatch):
    _workspace(tmp_path)
    seen: list[RunConfig] = []

    def fake_run_pack(pack, cfg, *, lmstudio, root):
        seen.append(cfg)
        raise RunRefused("stop here", exitCode=4)

    monkeypatch.setattr(cli, "LMStudio", lambda base_url: DecisionStub())
    monkeypatch.setattr(cli, "run_pack", fake_run_pack)
    code = cli.main(
        ["run", "--root", str(tmp_path), "--pack", _PACK_ID, "--candidate", _CANDIDATE_ID]
    )
    assert code == 4
    assert (seen[0].candidateId, seen[0].modelKey, seen[0].armKey) == (
        _CANDIDATE_ID, None, _CANDIDATE_ID
    )


def test_cli_run_candidate_end_to_end_stores_indexes_and_lists_the_candidate(
    tmp_path, monkeypatch, capsys
):
    """The done-condition through the CLI: `run --candidate` against a fake LM Studio stores a
    valid `model:decisions` record under `tmp_path`, `index rebuild` fills the candidate column,
    and `models --tested` lists the candidate by its `armKey`."""
    _workspace(tmp_path)
    stub = DecisionStub(decide_responses=_four(0.95, 0.1, 0.85, 0.9))
    monkeypatch.setattr(cli, "LMStudio", lambda base_url: stub)
    code = cli.main(
        ["run", "--root", str(tmp_path), "--pack", _PACK_ID, "--candidate", _CANDIDATE_ID]
    )
    assert code == cli.EXIT_OK, capsys.readouterr().err
    valid, invalid = load_history(tmp_path, packId=_PACK_ID)
    assert invalid == [] and [r.armKey for r in valid] == [_CANDIDATE_ID]

    rows = rebuild_index(tmp_path).read_text().splitlines()
    header = rows[0].split(",")
    cells = dict(zip(header, rows[1].split(",")))
    assert (cells["candidateId"], cells["handoffModelKey"], cells["modelKey"]) == (
        _CANDIDATE_ID, "", "laya"
    )
    capsys.readouterr()
    assert cli.main(["models", "--tested", "--root", str(tmp_path), "--pack", _PACK_ID]) == 0
    assert capsys.readouterr().out.split() == [_CANDIDATE_ID]


def _stored_candidate_run(root: Path) -> Any:
    _workspace(root)
    run, _ = _run(root, DecisionStub(decide_responses=_four(0.95, 0.1, 0.85, 0.9)))
    return run


def test_arm_key_is_the_candidate_id_on_a_candidate_run_and_the_model_key_otherwise(tmp_path):
    from conftest import run as chat_run

    candidate_run = _stored_candidate_run(tmp_path)
    assert (candidate_run.armKey, candidate_run.modelKey) == (_CANDIDATE_ID, "laya")
    chat = chat_run("r-chat")
    assert chat.armKey == chat.modelKey == "qwen/qwen3-4b-2507"
    reference = chat_run("r-bm25", arm_kind="deterministic")
    assert reference.armKey == reference.modelKey == "bm25"


def test_index_row_carries_the_candidate_and_hand_off_columns_empty_on_a_chat_arm(tmp_path):
    from conftest import run as chat_run

    candidate_row = _index_row(_stored_candidate_run(tmp_path), valid=True)
    assert (candidate_row["candidateId"], candidate_row["handoffModelKey"]) == (_CANDIDATE_ID, "")
    chat_row = _index_row(chat_run("r-chat"), valid=True)
    assert (chat_row["candidateId"], chat_row["handoffModelKey"]) == ("", "")


def test_index_row_carries_the_hand_off_model_key_when_the_fingerprint_has_one():
    from conftest import run as chat_run

    row = _index_row(chat_run("r-h", call_surface="handoff"), valid=True)
    assert (row["candidateId"], row["handoffModelKey"]) == (
        "guard-laya-band-v1", "qwen/qwen3-4b-2507"
    )


def test_compare_models_selects_a_candidate_arm_by_its_arm_key(tmp_path):
    """v1.3 §3.3: `_select_arms --models` keys on `armKey`, so a candidate is named by its id;
    its decision model's key does not name it."""
    from conftest import run as chat_run

    candidate_run = _stored_candidate_run(tmp_path)
    chat = chat_run("r-chat")
    arms = cli._select_arms(
        [chat, candidate_run],
        models=f"{chat.modelKey},{_CANDIDATE_ID}",
        session=None,
        negative_control=False,
    )
    assert arms == [chat, candidate_run]
    with pytest.raises(cli.UnknownModelKey):
        cli._select_arms([candidate_run], models="laya", session=None, negative_control=False)


def test_build_fingerprint_takes_decision_fields_iff_the_surface_is_decisions(tmp_path):
    """The `model:decisions` additions can neither go missing on a decisions run nor leak onto
    another profile."""
    from datetime import datetime, timezone

    from modelbench.runner import _build_fingerprint

    pack = load_pack(_FIXTURE_PACK)
    common = dict(
        pack=pack,
        model_info=_laya(loaded=4096),
        host={"attested": {}},
        started_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
        ended_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
        resident_at_start=[],
        resident_at_end=[],
        residency_source="lmstudio-api-v0",
        runtime_name=None,
        runtime_version=None,
        catalog_after_load=[_laya(loaded=4096)],
        prompt_config=pack.prompt_config(),
    )
    with pytest.raises(ValueError, match="iff call_surface"):
        _build_fingerprint(call_surface="decisions", decision_fields=None, **common)
    with pytest.raises(ValueError, match="iff call_surface"):
        _build_fingerprint(
            call_surface="embeddings", decision_fields={"candidateId": "c"}, **common
        )


def test_a_choice_decision_scores_its_chosen_value_with_the_confidence_as_probability(tmp_path):
    """v1.3 §3.3: `probability` is a predicate's p, or a choice's **confidence** — never one of
    its per-value probabilities. No choice role is built yet (U4), so the candidate is a loaded
    one re-shaped to a choice question and the scorer records what it is handed."""
    _workspace(tmp_path)
    loaded = Candidate.load(tmp_path, _CANDIDATE_ID)
    question = {
        "type": "choice",
        "name": "kind",
        "instructions": "",
        "choices": [{"value": "a", "instructions": ""}, {"value": "b", "instructions": ""}],
    }
    choice_candidate = replace(
        loaded,
        question=MappingProxyType(question),
        policy=MappingProxyType({"kind": "choice-confidence", "minConfidence": 0.0}),
    )
    answer = DecisionAnswer(
        name="kind", type="choice", probability=None, choice="b",
        probabilities=(("a", 0.3), ("b", 0.7)), confidence=0.64,
    )
    seen: list[tuple[Any, Any]] = []

    class Recorder:
        def decision_input(self, item_input, *, pack):
            return "x"

        def score_answer(self, item_input, answer, outcome, timing, *, pack, decided_by,
                         probability, failed_call, reason, input_tokens):
            seen.append((answer, probability))
            return _decision_item(item_input["itemId"], 1.0)

    pack = load_pack(tmp_path / "packs" / _PACK_ID)
    _drive_decision_items(
        pack,
        candidate=choice_candidate,
        scorer=Recorder(),
        lmstudio=DecisionStub(
            decide_responses=[
                DecisionResult(answers=(answer,), model="laya", usage=None, wallClockMs=5.0)
            ] * 4
        ),
        model_info=_laya(loaded=4096),
        baseline_residency=list(_LAYA_RESIDENT),
        clock=_Clock(0.01),
    )
    assert seen == [("b", 0.64)] * 4
