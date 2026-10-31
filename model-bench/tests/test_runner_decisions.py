"""typed-decision-models U1d.2 and U1d.3 — the plain `decisions` arm and its run-time deltas
(plan v1.3 §3.3, §4 U1d; plan2 §3.2–§3.4, §3.6; plan3 §3.3, §3.6).

Offline throughout: a hand-built `LMStudio`-shaped stub, an on-disk fixture pack
(`tests/fixtures/packs/decision_point_guard/`) and a candidate written under `tmp_path`. Nothing
here writes under `model-bench/results/`.

Sections, in the units' landing order: the scorer seam's three hooks; the packs half; the runner
(refusals, the per-item flow, `latency_block`, the fingerprint); the done-condition run; the CLI;
U1d.3's chat-timeout pin, kept decision failures, load rule and decision-call counts.
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


def test_the_fixture_decision_point_pack_fails_only_the_two_size_refusals_and_its_pad_validates(
    tmp_path,
):
    """The 4-item fixture is below v1.3 §3.5's contract on size alone — 4 `eval` items under
    `MIN_EVAL_ITEMS`, and no guard side check — so every other axis, and every other decision-
    point refusal, is clean. `run_pack` never validates, so its tests keep the 4 items; a `cli run`
    test, which validates first, pads its copy (`_pad_to_the_contract`) and that copy is clean."""
    from modelbench.packs import derive_decision_support, load_pack, validate_pack

    pack = load_pack(_FIXTURE_PACK)
    problems = validate_pack(pack)
    assert len(problems) == 2, problems
    assert "4 item(s) in stratum 'eval', fewer than MIN_EVAL_ITEMS = 200" in problems[0]
    assert "needs its 'side-check' stratum" in problems[1]
    assert derive_decision_support(pack.manifest["environment"]["requires"]) is True

    padded = load_pack(_workspace(tmp_path, contract_valid=True))
    assert validate_pack(padded) == []


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
    LMStudioUnreachable,
    LoadedInstance,
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


#: The default fit manifest: two calibration inputs disjoint from the fixture pack's items, as
#: the file format `decisions.load_fit_manifest` reads (sorted lowercase-hex lines, each ending
#: in `\n`). `_workspace` writes it and declares its real sha256 and entry count.
_MANIFEST_HASHES = tuple(
    sorted(hashlib.sha256(f"calibration-{n}".encode()).hexdigest() for n in (1, 2))
)


def _manifest_bytes(hashes: tuple[str, ...]) -> bytes:
    return "".join(h + "\n" for h in hashes).encode()


def _candidate_body(**overrides: Any) -> dict[str, Any]:
    manifest = _manifest_bytes(_MANIFEST_HASHES)
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
            "sha256": hashlib.sha256(manifest).hexdigest(),
            "size": len(_MANIFEST_HASHES),
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


#: A sentinel for `DecisionStub(instance_config=...)`: the instance carries no `config` key at all.
_NO_CONFIG_KEY = object()

#: laya's `loaded_instances[0].config` as the U2p probe read it after its explicit load
#: (`tests/fixtures/lmstudio/api_v1_load_probe.json`, `ws:agent-team` a93905a6…).
_U2P = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "lmstudio" / "api_v1_load_probe.json")
    .read_text()
)
_LAYA_CONFIG = _U2P["laya_instance"]["config"]


class DecisionStub:
    """An `LMStudio`-shaped stub for the `decisions` surface: `decide` pops results in call order
    (an exception instance is raised), `residency` pops snapshots (default: laya resident),
    `catalog` returns the pre-load list once, then the post-load list.

    The explicit load (plan3 §3.7): `load_model` records the call (or raises `load_failure`) and
    marks the model loaded; `loaded_instances` answers `GET /api/v1/models` from that state —
    one instance carrying `instance_config` for a loaded model, `()` otherwise — so a read made
    before the load sees no instance. `loaded` starts as `initially_loaded` (default: laya, to
    match the default residency). `order` logs loads, warm-ups and instance reads in sequence.

    The decisions `warm_up` returns `warm_up_decision` on `LoadResult.decision`, as the real one
    does (plan2 §3.4). Once `decide`'s queue is empty it answers with that same decision, so the
    within-load re-check after the last item agrees unless a test queues a fifth response."""

    def __init__(
        self,
        *,
        decide_responses: list[Any] | None = None,
        residency_sequence: list[list[ResidentModel]] | None = None,
        catalog_before: list[ModelInfo] | None = None,
        catalog_after: list[ModelInfo] | None = None,
        warm_up_decision: DecisionResult | None = None,
        initially_loaded: tuple[str, ...] = ("laya",),
        instance_config: Any = _LAYA_CONFIG,
        load_failure: BaseException | None = None,
    ) -> None:
        self.loaded = set(initially_loaded)
        self.instance_config = instance_config
        self.load_failure = load_failure
        self.load_calls: list[dict[str, Any]] = []
        self.was_resident_before: list[bool] = []
        self.order: list[tuple[str, str]] = []
        self._decide = list(decide_responses or [])
        self.warm_up_decision = warm_up_decision if warm_up_decision is not None else _decided(0.5)
        #: Optional: one decision per warm-up call, in order (the mandatory one, then extras).
        self.warm_up_sequence: list[DecisionResult] = []
        #: Every residency probe and decide call, in order.
        self.events: list[Any] = []
        self._residency = list(residency_sequence or [])
        self._catalogs = [
            catalog_before if catalog_before is not None else [_laya(), _qwen()],
            catalog_after if catalog_after is not None else [_laya(loaded=4096), _qwen()],
        ]
        self.decide_calls: list[dict[str, Any]] = []
        self.warm_up_calls: list[dict[str, Any]] = []
        self.chat_calls: list[Any] = []
        self.chat_reply: Any = None

    def probe(self) -> str:
        return "api-v0"

    def residency(self) -> list[ResidentModel]:
        """Pops the next snapshot; an exception instance in the sequence is raised instead."""
        self.events.append("residency")
        snapshot = self._residency.pop(0) if self._residency else list(_LAYA_RESIDENT)
        if isinstance(snapshot, BaseException):
            raise snapshot
        return snapshot

    def catalog(self) -> list[ModelInfo]:
        return list(self._catalogs.pop(0) if len(self._catalogs) > 1 else self._catalogs[0])

    def load_model(self, model, *, timeout_s):
        self.load_calls.append({"model": model, "timeout_s": timeout_s})
        self.order.append(("load_model", model))
        if self.load_failure is not None:
            raise self.load_failure
        self.loaded.add(model)
        return LoadedInstance(
            instance_id=model, status="loaded", type="decision", load_time_seconds=1.56,
            load_config=self.instance_config,
        )

    def loaded_instances(self, model_key):
        self.order.append(("loaded_instances", model_key))
        if model_key not in self.loaded:
            return ()
        instance = {"id": model_key}
        if self.instance_config is not _NO_CONFIG_KEY:
            instance["config"] = self.instance_config
        return (instance,)

    def warm_up(
        self, model_key, *, call_surface, system_prompt, was_resident_before, timeout_s,
        question=None,
    ):
        self.order.append(("warm_up", model_key))
        self.was_resident_before.append(was_resident_before)
        self.warm_up_calls.append(
            {"model": model_key, "call_surface": call_surface, "question": question,
             "timeout_s": timeout_s, "system_prompt": system_prompt}
        )
        runtime = (
            {"name": "llama.cpp", "version": "1.52.0"} if call_surface == "chat" else None
        )
        return LoadResult(
            wallClockMs=2500.0,
            wasResidentBefore=was_resident_before,
            runtime=runtime,
            stats=None,
            decision=(
                (self.warm_up_sequence.pop(0) if self.warm_up_sequence else self.warm_up_decision)
                if call_surface == "decisions"
                else None
            ),
        )

    def decide(self, input, questions, *, model, timeout_s):
        self.decide_calls.append(
            {"input": input, "questions": questions, "model": model, "timeout_s": timeout_s}
        )
        self.events.append(("decide", input))
        response = self._decide.pop(0) if self._decide else self.warm_up_decision
        if isinstance(response, BaseException):
            raise response
        return response

    def chat(self, messages, *, model, temperature, max_tokens, timeout_s, tools=None):
        """Only a chat arm on the fixture pack reaches here (`chat_reply` set); the decisions
        surface never chats."""
        self.chat_calls.append({"model": model, "timeout_s": timeout_s})
        if self.chat_reply is None:
            raise AssertionError("the decisions surface made a chat call")
        return self.chat_reply


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


def _pad_to_the_contract(pack_dir: Path) -> None:
    """U3b.1: `cli run` validates first, and v1.3 §3.5 refuses the 4-item fixture (fewer than
    `MIN_EVAL_ITEMS` `eval` items, no guard side check). Appends synthetic `eval` items, each with
    its own condition so no input hash repeats, up to the floor, then one human-curated
    `side-check` item. The fixture's own four items stay first, in order."""
    from modelbench.decisions import MIN_EVAL_ITEMS

    path = pack_dir / "items.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    evals = sum(1 for row in rows if row.get("stratum") == "eval")
    for n in range(MIN_EVAL_ITEMS - evals):
        rows.append({
            "itemId": f"pad-{n:03d}", "tier": "clear_suspend", "path": "understanding",
            "stratum": "eval", "origin": "synthetic", "condition": f"padding condition {n}",
            "understanding": {"request": "pad"}, "turns": [], "expected": False,
        })
    rows.append({
        "itemId": "sc-01", "tier": "clear_advance", "path": "understanding",
        "stratum": "side-check", "origin": "human", "condition": "the side check's condition",
        "understanding": {"request": "checkout returns 502"}, "turns": [], "expected": True,
    })
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _workspace(
    root: Path,
    *,
    manifest_edit=None,
    items_edit=None,
    fit_manifest_bytes=None,
    contract_valid: bool = False,
    **candidate,
) -> Path:
    """`root` laid out as a model-bench root: the fixture pack under `packs/`, `host.json`, and
    the candidate and its fit manifest under `candidates/`. Never the shipped `model-bench/`
    root. `fit_manifest_bytes` replaces the default manifest file's content (its declaration is
    the candidate body's, so a test that changes one decides about the other). `contract_valid`
    pads the pack copy to v1.3 §3.5's contract (`_pad_to_the_contract`), for a `cli run` test."""
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
    if contract_valid:
        _pad_to_the_contract(pack_dir)
    _host_json(root)
    (root / "candidates").mkdir()
    (root / "candidates" / f"{_CANDIDATE_ID}.json").write_text(
        json.dumps(_candidate_body(**candidate), indent=2)
    )
    (root / "candidates" / f"{_CANDIDATE_ID}.fit.txt").write_bytes(
        fit_manifest_bytes if fit_manifest_bytes is not None else _manifest_bytes(_MANIFEST_HASHES)
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
    # U3b.1: each eval item declares its per-kind verdict metric and the pooled one (v1.3 §3.5).
    assert counts == {
        "fx-cs-01": {"falseAdvanceRateByUnderstanding": 1, "falseAdvanceRate": 1},
        "fx-cs-02": {"falseAdvanceRateByTurns": 0, "falseAdvanceRate": 0},
        "fx-ca-01": {"falseSuspendRateByUnderstanding": 0, "falseSuspendRate": 0},
        "fx-ca-02": {"falseSuspendRateByTurns": 1, "falseSuspendRate": 1},
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
    _run(tmp_path, lms, requestTimeoutSeconds=180.0, firstCallTimeoutSeconds=300.0)
    pack = load_pack(pack_dir)
    expected_inputs = [classification.decision_input(i, pack=pack) for i in pack.iter_items()]
    # Four items, then the within-load re-check (plan2 §3.4) with the warm-up's input.
    assert [c["input"] for c in lms.decide_calls] == [*expected_inputs, "warm-up"]
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


def test_a_failed_decision_call_is_kept_timed_by_the_item_clock_and_scored_as_failed(tmp_path):
    """Plan2 §3.3: a decision-call timeout or failure keeps the item — outcome `"fail"`, the item
    clock at the raise (250 ms here; the failed call has no clock of its own), no completed call,
    `withheldFor=None`, `failedCall="decision"`, nothing scoreable. The run is stored, not
    refused."""
    _workspace(tmp_path)
    responses = [
        _decided(0.9),
        LMStudioCallTimeout("timed out"),
        LMStudioCallFailed("HTTP 500", status=500, body="boom"),
        _decided(0.1),
    ]
    run, _ = _run(tmp_path, DecisionStub(decide_responses=responses), clock=_Clock(0.25))
    kept = run.items[1:3]
    for item in kept:
        assert item.outcome == "fail"
        assert item.timing.wallClockMs == pytest.approx(250.0)
        assert (item.timing.calls, item.timing.withheldFor) == ((), None)
        assert item.latencyMs == pytest.approx(250.0)
        assert (item.scoreable, item.counts) == ({}, {})
        assert item.detail == {
            "path": item.detail["path"],
            "tier": item.detail["tier"],
            "failedCall": "decision",
            "probability": None,
            "inputTokens": None,
        }
    assert [i.detail.get("decidedBy") for i in run.items] == [
        "decision-model", None, None, "decision-model"
    ]
    path = store(run, tmp_path)
    valid, invalid = load_history(tmp_path, packId=_PACK_ID)
    assert invalid == [] and [r.runId for r in valid] == [run.runId], path


def test_a_kept_decision_failure_after_a_snapshot_without_the_model_is_withheld_for_load(tmp_path):
    _workspace(tmp_path)
    residency = [[], list(_LAYA_RESIDENT), list(_LAYA_RESIDENT), []]  # item 2's probe: gone
    lms = DecisionStub(
        decide_responses=[_decided(0.9), LMStudioCallTimeout("t"), _decided(0.9), _decided(0.9)],
        residency_sequence=residency,
    )
    run, _ = _run(tmp_path, lms)
    item = run.items[1]
    assert (item.timing.withheldFor, item.detail["failedCall"]) == ("load", "decision")
    assert item.timing.wallClockMs == pytest.approx(250.0) and item.latencyMs is None


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
        # measured from the manifest file `_workspace` wrote: its two lines, hashed by hand with
        # `printf '%s\n%s\n' <h1> <h2> | sha256sum`
        "fitManifestHash": "3c334d5b1aa6b88abd917bde5d2c75d6c7a0bc11628a7be39ecf9cb02b0e47e7",
        "fitManifestSize": 2,
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
    _workspace(tmp_path, contract_valid=True)
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
    _workspace(tmp_path, contract_valid=True)
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
        handoff_fields=None,
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


# ==================================================================================================
# 6. U1d.3 — the run-time deltas (plan2 §3.2–§3.4, §3.6; plan3 §3.3, §3.6)
# ==================================================================================================

from modelbench.lmstudio import ChatResult  # noqa: E402
from modelbench.results import RunResult  # noqa: E402

# --- the chat-call timeout pin (plan2 §3.2) -------------------------------------------------------


def test_an_explicit_request_timeout_other_than_180_refuses_a_decision_point_run_up_front(
    tmp_path,
):
    for explicit in (120.0, 179.9, 600.0):
        root = tmp_path / f"t{explicit:g}"
        root.mkdir()
        _workspace(root)
        lms = DecisionStub(decide_responses=_four(0.9, 0.1, 0.9, 0.9))
        with pytest.raises(RunRefused) as caught:
            _run(root, lms, requestTimeoutSeconds=explicit)
        assert caught.value.exitCode == 4, explicit
        assert "--request-timeout" in str(caught.value) and "180" in str(caught.value)
        assert lms.warm_up_calls == [] and lms.decide_calls == [], "refused after a model call"


def test_cli_run_request_timeout_120_on_a_decision_point_pack_exits_4(
    tmp_path, monkeypatch, capsys
):
    """On a contract-valid pack, so the exit 4 is the timeout refusal's, not `validate_pack`'s."""
    _workspace(tmp_path, contract_valid=True)
    stub = DecisionStub(decide_responses=_four(0.9, 0.1, 0.9, 0.9))
    monkeypatch.setattr(cli, "LMStudio", lambda base_url: stub)
    base = ["run", "--root", str(tmp_path), "--pack", _PACK_ID, "--candidate", _CANDIDATE_ID]
    assert cli.main([*base, "--request-timeout", "120"]) == 4
    assert "--request-timeout" in capsys.readouterr().err
    assert stub.warm_up_calls == [] and stub.decide_calls == []
    assert not (tmp_path / "results").exists()


def test_a_decision_point_run_with_no_flag_or_an_explicit_180_records_180(tmp_path):
    for explicit in (None, 180.0):
        root = tmp_path / str(explicit)
        root.mkdir()
        _workspace(root)
        run, _ = _run(
            root,
            DecisionStub(decide_responses=_four(0.9, 0.1, 0.9, 0.9)),
            requestTimeoutSeconds=explicit,
        )
        assert run.requestTimeoutSeconds == 180.0
        assert RunResult.from_dict(run.to_dict()).requestTimeoutSeconds == 180.0


def test_a_chat_arm_on_a_decision_point_pack_with_no_flag_chats_at_180(tmp_path):
    """The pin's point: the incumbent's chat calls on a decision-point pack run at 180 s, not at
    the 120 s default, when no flag is given."""
    _workspace(tmp_path)
    qwen_resident = [ResidentModel(id="qwen/qwen3-4b-2507", state="loaded")]
    lms = DecisionStub(residency_sequence=[list(qwen_resident) for _ in range(8)])
    lms.chat_reply = ChatResult(
        message={"content": '{"decision": false}'},
        tool_calls=(),
        toolCallForm="prose",
        stats=None,
        model_info=None,
        runtime=None,
        usage=None,
        wallClockMs=900.0,
    )
    run, _ = _run(tmp_path, lms, candidateId=None, modelKey="qwen/qwen3-4b-2507")
    assert [c["timeout_s"] for c in lms.chat_calls] == [180.0] * 4
    assert run.requestTimeoutSeconds == 180.0


def test_a_run_off_a_decision_point_pack_records_its_request_timeout(tmp_path, monkeypatch):
    """Every run records it: the default 120 with no flag, the flag's value with one."""
    from test_runner import (
        FakeItemScorer,
        StubLMStudio,
        _item_pack,
        chat_result_with_stats,
        model_info,
        resident,
    )
    from test_runner import _host_json as chat_host_json

    monkeypatch.setattr("modelbench.runner._load_item_scorer", lambda pack: FakeItemScorer())
    chat_host_json(tmp_path)
    for explicit, recorded in ((None, 120.0), (45.0, 45.0)):
        lms = StubLMStudio(
            catalog_result=[model_info()],
            chat_responses=[chat_result_with_stats(wall_clock_ms=100.0)],
            residency_sequence=[resident(), resident(), resident(), resident()],
        )
        cfg = RunConfig(
            modelKey="qwen/qwen3-4b-2507", sessionId=None, referenceKey=None,
            requestTimeoutSeconds=explicit,
        )
        run, _ = run_pack(_item_pack(), cfg, lmstudio=lms, root=tmp_path)
        assert run.requestTimeoutSeconds == recorded, explicit


def test_a_legacy_record_without_request_timeout_loads_it_as_none():
    shipped = Path(__file__).resolve().parents[1]
    records = sorted((shipped / "results" / "runs").glob("*.json"))
    assert records
    for path in records:
        body = json.loads(path.read_text())
        assert "requestTimeoutSeconds" not in body, path.name
        assert RunResult.from_dict(body).requestTimeoutSeconds is None, path.name


# --- the load rule, the within-load re-check and `basis` (plan2 §3.4) ----------------------------


def _basis_run(root: Path, lms: DecisionStub, **cfg: Any):
    _workspace(root, decisionTimeoutSeconds=1.5)
    return _run(root, lms, **cfg)


def test_an_equal_re_check_after_the_last_item_records_by_construction(tmp_path):
    lms = DecisionStub(decide_responses=_four(0.9, 0.1, 0.9, 0.9))
    run, _ = _basis_run(tmp_path, lms)
    assert (run.designEffect, run.basis) == (1.0, "by-construction")
    recheck = lms.decide_calls[-1]
    assert recheck == {
        "input": "warm-up", "questions": [_QUESTION], "model": "laya", "timeout_s": 1.5
    }
    # After the last item: the load rule's probe, the re-check, then step 10's end capture.
    assert lms.events[-4:] == [
        ("decide", lms.decide_calls[3]["input"]), "residency", ("decide", "warm-up"), "residency"
    ]


def test_a_re_check_with_a_different_probability_records_assumed(tmp_path):
    lms = DecisionStub(decide_responses=[*_four(0.9, 0.1, 0.9, 0.9), _decided(0.5000001)])
    run, _ = _basis_run(tmp_path, lms)
    assert run.basis == "assumed" and len(lms.decide_calls) == 5


def test_a_re_check_that_raises_records_assumed(tmp_path):
    for n, failure in enumerate((
        LMStudioCallTimeout("timed out"),
        LMStudioCallFailed("HTTP 500", status=500, body="boom"),
        LMStudioCallFailed("HTTP 500", status=500, body="input is too large to process"),
    )):
        root = tmp_path / str(n)
        root.mkdir()
        lms = DecisionStub(decide_responses=[*_four(0.9, 0.1, 0.9, 0.9), failure])
        run, _ = _basis_run(root, lms)
        assert run.basis == "assumed", failure
        assert len(run.items) == 4  # the run is stored, not refused


def test_an_item_withheld_for_load_records_assumed_with_no_re_check_call(tmp_path):
    residency = [[], list(_LAYA_RESIDENT), list(_LAYA_RESIDENT), []]  # item 2's probe: gone
    lms = DecisionStub(decide_responses=_four(0.9, 0.1, 0.9, 0.9), residency_sequence=residency)
    run, _ = _basis_run(tmp_path, lms)
    assert [i.timing.withheldFor for i in run.items] == [None, "load", None, None]
    assert run.basis == "assumed"
    assert [c["input"] for c in lms.decide_calls].count("warm-up") == 0
    assert len(lms.decide_calls) == 4


def test_the_model_absent_at_the_end_probe_records_assumed_with_no_re_check_call(tmp_path):
    residency = [
        [], list(_LAYA_RESIDENT),          # start, baseline
        *[list(_LAYA_RESIDENT)] * 4,       # before each item
        [],                                # the load rule's probe: laya gone
    ]
    lms = DecisionStub(decide_responses=_four(0.9, 0.1, 0.9, 0.9), residency_sequence=residency)
    run, _ = _basis_run(tmp_path, lms)
    assert [i.timing.withheldFor for i in run.items] == [None] * 4
    assert run.basis == "assumed"
    assert len(lms.decide_calls) == 4


def test_the_re_check_compares_with_the_mandatory_warm_up_never_a_warmup_extra(tmp_path):
    """`--warmup 1`: the extra warm-up answers 0.7, the mandatory one 0.5. A re-check of 0.5 is
    one load; a re-check of 0.7 is not."""
    for recheck_p, basis in ((0.5, "by-construction"), (0.7, "assumed")):
        root = tmp_path / str(recheck_p)
        root.mkdir()
        lms = DecisionStub(decide_responses=[*_four(0.9, 0.1, 0.9, 0.9), _decided(recheck_p)])
        lms.warm_up_sequence = [_decided(0.5), _decided(0.7)]
        run, _ = _basis_run(root, lms, warmupExtra=1)
        assert len(lms.warm_up_calls) == 2
        assert run.basis == basis, recheck_p


def test_an_assumed_run_stores_and_reads_back_valid(tmp_path):
    lms = DecisionStub(decide_responses=[*_four(0.9, 0.1, 0.9, 0.9), _decided(0.1)])
    run, _ = _basis_run(tmp_path, lms)
    store(run, tmp_path)
    valid, invalid = load_history(tmp_path, packId=_PACK_ID)
    assert invalid == [] and [r.basis for r in valid] == ["assumed"]


def _choice(choice: str, confidence: float, probs=(("a", 0.3), ("b", 0.7))) -> DecisionAnswer:
    return DecisionAnswer(
        name="kind", type="choice", probability=None, choice=choice, probabilities=probs,
        confidence=confidence,
    )


def test_re_check_equality_is_exact_on_the_fields_plan2_names():
    from modelbench.runner import _same_decision

    p = _decided(0.5).answers[0]
    assert _same_decision(p, _decided(0.5).answers[0])
    assert not _same_decision(p, _decided(0.5 + 1e-12).answers[0])
    assert _same_decision(_choice("b", 0.64), _choice("b", 0.64))
    # per-value probabilities are not compared; choice and confidence are
    assert _same_decision(_choice("b", 0.64), _choice("b", 0.64, probs=(("a", 0.2), ("b", 0.8))))
    assert not _same_decision(_choice("b", 0.64), _choice("a", 0.64))
    assert not _same_decision(_choice("b", 0.64), _choice("b", 0.65))
    assert not _same_decision(p, _choice("b", 0.5))


# --- the decision-call counts (plan2 §3.3 "Counts"; plan3 §3.6) -----------------------------------


def _metric(run, name: str):
    found = [m for m in run.aggregates.perClass if m.name == name]
    assert len(found) == 1, (name, [m.name for m in run.aggregates.perClass])
    return found[0]


def test_decision_call_failures_counts_the_injected_failures_over_every_item(tmp_path):
    _workspace(tmp_path)
    responses = [
        LMStudioCallTimeout("t"),
        _decided(0.9),
        LMStudioCallFailed("HTTP 500", status=500, body="boom"),
        _decided(0.1),
    ]
    run, _ = _run(tmp_path, DecisionStub(decide_responses=responses))
    m = _metric(run, "decisionCallFailures")
    assert (m.successes, m.n, m.unit) == (2, 4, "item")
    store(run, tmp_path)
    valid, invalid = load_history(tmp_path, packId=_PACK_ID)
    assert invalid == []
    assert _metric(valid[0], "decisionCallFailures") == m


def test_decision_call_failures_is_zero_on_a_clean_run_and_never_a_verdict_metric(tmp_path):
    _workspace(tmp_path)
    run, _ = _run(tmp_path, DecisionStub(decide_responses=_four(0.9, 0.1, 0.9, 0.9)))
    m = _metric(run, "decisionCallFailures")
    assert (m.successes, m.n) == (0, 4)
    pack = load_pack(tmp_path / "packs" / _PACK_ID)
    assert "decisionCallFailures" not in pack.manifest["metrics"]["verdictMetrics"]


def test_a_chat_arm_on_a_decision_point_pack_carries_no_decision_call_counts(tmp_path):
    _workspace(tmp_path)
    qwen_resident = [ResidentModel(id="qwen/qwen3-4b-2507", state="loaded")]
    lms = DecisionStub(residency_sequence=[list(qwen_resident) for _ in range(8)])
    lms.chat_reply = ChatResult(
        message={"content": '{"decision": false}'}, tool_calls=(), toolCallForm="prose",
        stats=None, model_info=None, runtime=None, usage=None, wallClockMs=900.0,
    )
    run, _ = _run(tmp_path, lms, candidateId=None, modelKey="qwen/qwen3-4b-2507")
    names = {m.name for m in run.aggregates.perClass}
    assert not names & {"decisionCallFailures", "overLimit"}


_OVER_LIMIT_ROWS = {
    row["case"]: row
    for row in map(
        json.loads,
        (Path(__file__).resolve().parent / "fixtures" / "decision-contract" / "over-limit.jsonl")
        .read_text()
        .splitlines(),
    )
}


def _failure(case: str) -> LMStudioCallFailed:
    row = _OVER_LIMIT_ROWS[case]
    return LMStudioCallFailed(f"HTTP {row['status']}", status=row["status"], body=row["body"])


def test_an_over_limit_decision_call_is_a_kept_failure_with_reason_over_limit(tmp_path):
    """Plan3 §3.3, `decisions` surface: the kept decision failure plus `reason="over-limit"`; it
    counts in `overLimit` and not in `decisionCallFailures` (plan3 §3.6). The body is the real
    LM Studio over-limit body (`over-limit.jsonl`)."""
    _workspace(tmp_path)
    responses = [
        _decided(0.9), _failure("body-document-at-500"), _decided(0.1), _decided(0.9)
    ]
    run, _ = _run(tmp_path, DecisionStub(decide_responses=responses))
    item = run.items[1]
    assert item.outcome == "fail" and (item.scoreable, item.counts) == ({}, {})
    assert item.timing.wallClockMs == pytest.approx(250.0)
    assert (item.timing.calls, item.timing.withheldFor) == ((), None)
    assert item.detail == {
        "path": item.detail["path"],
        "tier": item.detail["tier"],
        "failedCall": "decision",
        "probability": None,
        "inputTokens": None,
        "reason": "over-limit",
    }
    assert [i.detail.get("reason") for i in run.items] == [None, "over-limit", None, None]
    over, failures = _metric(run, "overLimit"), _metric(run, "decisionCallFailures")
    assert (over.successes, over.n, over.unit) == (1, 4, "item")
    assert (failures.successes, failures.n) == (0, 4)


def test_a_failure_that_is_not_over_limit_stays_a_plain_decision_failure(tmp_path):
    """Plan3 §3.3: "Any other 500 stays a failure" — and so does the marker at 400, or
    upper-cased. Each counts in `decisionCallFailures`, never in `overLimit`, and carries no
    `reason`."""
    for case in ("another-500-body", "body-document-at-400", "marker-upper-cased-at-500",
                 "empty-body-at-500", "null-body-at-500"):
        root = tmp_path / case
        root.mkdir()
        _workspace(root)
        responses = [_decided(0.9), _failure(case), _decided(0.1), _decided(0.9)]
        run, _ = _run(root, DecisionStub(decide_responses=responses))
        item = run.items[1]
        assert item.detail["failedCall"] == "decision" and "reason" not in item.detail, case
        assert _metric(run, "overLimit").successes == 0, case
        assert _metric(run, "decisionCallFailures").successes == 1, case


def test_over_limit_and_decision_failures_count_apart_in_one_run(tmp_path):
    _workspace(tmp_path)
    responses = [
        _failure("body-document-at-500"),
        LMStudioCallTimeout("t"),
        _failure("another-500-body"),
        _failure("body-document-at-500"),
    ]
    run, _ = _run(tmp_path, DecisionStub(decide_responses=responses))
    assert (_metric(run, "overLimit").successes, _metric(run, "decisionCallFailures").successes) \
        == (2, 2)
    assert all(i.detail["failedCall"] == "decision" for i in run.items)


def test_over_limit_is_zero_over_n_on_a_clean_run(tmp_path):
    """Plan3 §3.6: `overLimit 0/n` is still recorded, so a report can print it."""
    _workspace(tmp_path)
    run, _ = _run(tmp_path, DecisionStub(decide_responses=_four(0.9, 0.1, 0.9, 0.9)))
    m = _metric(run, "overLimit")
    assert (m.successes, m.n, m.unit) == (0, 4, "item")
    assert [m.name for m in run.aggregates.perClass][-2:] == ["decisionCallFailures", "overLimit"]


# ==================================================================================================
# 7. `rank`'s arm dedupe keys on `armKey` (impl review Pass 6, P6-n1)
# ==================================================================================================


def test_rank_keeps_two_candidates_on_one_decision_model_and_dedupes_by_arm_key():
    """Two candidates built on `laya` share a `modelKey`; deduping on it would silently drop the
    older one. Keyed on `armKey`, both stay, and a repeated candidate keeps its newest run."""
    from conftest import decisions_fields
    from conftest import run as stored_run

    def candidate_run(run_id: str, candidate_id: str):
        return stored_run(
            run_id,
            call_surface="decisions",
            fingerprint_fields=decisions_fields(candidateId=candidate_id),
        )

    a1, b, a2 = candidate_run("r1", "cand-a"), candidate_run("r2", "cand-b"), candidate_run(
        "r3", "cand-a"
    )
    assert a1.modelKey == b.modelKey == a2.modelKey == "laya"
    assert cli._select_rank_arms([a1, b, a2], session=None) == [a2, b]


# ==================================================================================================
# 8. U2.1 — explicit loading (plan3 §3.7): the load step, co-residency, `decisionInstanceConfig`
# ==================================================================================================

_QWEN_RESIDENT = [ResidentModel(id="qwen/qwen3-4b-2507", state="loaded")]


def _absent_at_start(**kw: Any) -> DecisionStub:
    """laya absent from `residentModelsAtStart` and not loaded; resident from the baseline on."""
    return DecisionStub(
        decide_responses=_four(0.9, 0.1, 0.9, 0.9),
        residency_sequence=[[], *[list(_LAYA_RESIDENT)] * 7],
        initially_loaded=(),
        **kw,
    )


def test_a_decision_model_absent_at_start_is_loaded_once_before_the_warm_up(tmp_path):
    _workspace(tmp_path)
    lms = _absent_at_start()
    run, _ = _run(tmp_path, lms, firstCallTimeoutSeconds=240.0)
    assert lms.load_calls == [{"model": "laya", "timeout_s": 240.0}]
    assert lms.order[:2] == [("load_model", "laya"), ("warm_up", "laya")]
    assert lms.was_resident_before == [True]  # the warm-up found it loaded, so it is no load
    assert run.fingerprint.get("residentModelsAtStart") == []


def test_a_decision_model_resident_at_start_is_used_as_found_with_no_load(tmp_path):
    _workspace(tmp_path)
    lms = DecisionStub(decide_responses=_four(0.9, 0.1, 0.9, 0.9))
    _run(tmp_path, lms)
    assert lms.load_calls == []
    assert ("load_model", "laya") not in lms.order


@pytest.mark.parametrize(
    "failure",
    [
        LMStudioCallTimeout("timed out"),
        LMStudioCallFailed("HTTP 404", status=404, body='{"error": "model not found"}'),
        LMStudioUnreachable("no response"),
    ],
    ids=["timeout", "refused", "unreachable"],
)
def test_a_failed_load_refuses_with_exit_3_before_any_warm_up(tmp_path, failure):
    _workspace(tmp_path)
    lms = _absent_at_start(load_failure=failure)
    with pytest.raises(RunRefused) as caught:
        _run(tmp_path, lms)
    assert caught.value.exitCode == 3 and "'laya'" in str(caught.value)
    assert lms.warm_up_calls == [] and lms.decide_calls == []
    assert not (tmp_path / "results").exists()


def test_the_decision_instance_config_is_read_after_the_load_and_recorded_verbatim(tmp_path):
    """The read before the load would find no instance (a refusal); after it, laya's config."""
    _workspace(tmp_path)
    lms = _absent_at_start()
    run, _ = _run(tmp_path, lms)
    assert run.fingerprint.get("decisionInstanceConfig") == _LAYA_CONFIG
    assert run.fingerprint.get("decisionInstanceConfig")["physical_batch_size"] == 512
    reads = [i for i, e in enumerate(lms.order) if e == ("loaded_instances", "laya")]
    assert len(reads) == 1 and reads[0] > lms.order.index(("load_model", "laya"))
    store(run, tmp_path)
    valid, invalid = load_history(tmp_path, packId=_PACK_ID)
    assert invalid == [] and valid[0].fingerprint.get("decisionInstanceConfig") == _LAYA_CONFIG


def test_no_instance_listed_for_a_resident_decision_model_refuses_with_exit_3(tmp_path):
    """Impl review Pass 8, P8-M1's stub: `residency()` shows laya loaded (so no load step runs and
    co-residency passes) while `GET /api/v1/models` lists no instance under its key — an eviction
    between steps 7 and 8, or an `/api/v0` id that is not the `/api/v1` key. That is "not
    captured", never a `None` answer: the run refuses before any item and stores nothing."""
    _workspace(tmp_path)
    lms = DecisionStub(decide_responses=_four(0.9, 0.1, 0.9, 0.9), initially_loaded=())
    with pytest.raises(RunRefused) as caught:
        _run(tmp_path, lms)
    assert caught.value.exitCode == 3 and "no loaded instance" in str(caught.value)
    assert ("loaded_instances", "laya") in lms.order
    assert lms.decide_calls == []
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    "config", [_NO_CONFIG_KEY, None, ["context_length", 8192]],
    ids=["no-config-key", "config-null", "config-not-an-object"],
)
def test_an_instance_with_no_config_object_refuses_with_exit_3(tmp_path, config):
    """P8-M1: LM Studio's `config` is required on a loaded instance, so an instance without a
    `config` mapping is an anomaly, refused before any item — never recorded as `None`."""
    _workspace(tmp_path)
    lms = DecisionStub(decide_responses=_four(0.9, 0.1, 0.9, 0.9), instance_config=config)
    with pytest.raises(RunRefused) as caught:
        _run(tmp_path, lms)
    assert caught.value.exitCode == 3 and "no 'config' object" in str(caught.value)
    assert lms.decide_calls == []
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    "failure",
    [LMStudioUnreachable("no response"), LMStudioCallFailed("HTTP 404", status=404, body=None)],
    ids=["unreachable", "refused"],
)
def test_a_failed_instance_config_read_refuses_with_exit_3_before_any_item(
    tmp_path, monkeypatch, failure
):
    _workspace(tmp_path)
    lms = DecisionStub(decide_responses=_four(0.9, 0.1, 0.9, 0.9))

    def broken(model_key):
        raise failure

    monkeypatch.setattr(lms, "loaded_instances", broken)
    with pytest.raises(RunRefused) as caught:
        _run(tmp_path, lms)
    assert caught.value.exitCode == 3 and "instance config" in str(caught.value)
    assert lms.decide_calls == []


def test_a_decision_model_absent_after_its_warm_up_refuses_with_exit_3_before_any_item(tmp_path):
    """Co-residency on the `decisions` surface: one model, so the rule reduces to the decision
    model being resident after its warm-up (step 7's snapshot)."""
    _workspace(tmp_path)
    lms = DecisionStub(
        decide_responses=_four(0.9, 0.1, 0.9, 0.9),
        residency_sequence=[list(_LAYA_RESIDENT), list(_QWEN_RESIDENT)],  # start, baseline
    )
    with pytest.raises(RunRefused) as caught:
        _run(tmp_path, lms)
    assert caught.value.exitCode == 3 and "co-residency" in str(caught.value)
    assert lms.decide_calls == [] and len(lms.warm_up_calls) == 1
    assert not (tmp_path / "results").exists()


def _chat_arm_stub(residency: list[list[ResidentModel]]) -> DecisionStub:
    lms = DecisionStub(residency_sequence=residency, initially_loaded=())
    lms.chat_reply = ChatResult(
        message={"content": '{"decision": false}'}, tool_calls=(), toolCallForm="prose",
        stats=None, model_info=None, runtime=None, usage=None, wallClockMs=900.0,
    )
    return lms


def test_a_chat_arm_on_a_decision_point_pack_loads_its_absent_model_too(tmp_path):
    """Plan3 §3.7: "The arm's models are the chat model, the decision model, or both." The chat
    arm records no `decisionInstanceConfig` (it is a decision-profile field only)."""
    _workspace(tmp_path)
    lms = _chat_arm_stub([[], *[list(_QWEN_RESIDENT)] * 7])
    run, _ = _run(tmp_path, lms, candidateId=None, modelKey="qwen/qwen3-4b-2507")
    assert lms.load_calls == [{"model": "qwen/qwen3-4b-2507", "timeout_s": 300.0}]
    qwen = "qwen/qwen3-4b-2507"
    assert lms.order[:2] == [("load_model", qwen), ("warm_up", qwen)]
    assert run.fingerprint.get("decisionInstanceConfig", "ABSENT") == "ABSENT"
    assert ("loaded_instances", "qwen/qwen3-4b-2507") not in lms.order

    resident_root = tmp_path / "resident"
    resident_root.mkdir()
    _workspace(resident_root)
    lms = _chat_arm_stub([list(_QWEN_RESIDENT) for _ in range(8)])
    _run(resident_root, lms, candidateId=None, modelKey="qwen/qwen3-4b-2507")
    assert lms.load_calls == []


def test_a_pack_without_a_decision_point_block_never_loads(tmp_path, monkeypatch):
    """The load step is a decision-point pack's only: an ordinary chat run, its model absent at
    start, is JIT-loaded by its warm-up exactly as before (`StubLMStudio` has no `load_model`)."""
    from test_runner import (
        FakeItemScorer,
        StubLMStudio,
        _item_pack,
        chat_result_with_stats,
        model_info,
        resident,
    )
    from test_runner import _host_json as chat_host_json

    monkeypatch.setattr("modelbench.runner._load_item_scorer", lambda pack: FakeItemScorer())
    chat_host_json(tmp_path)
    lms = StubLMStudio(
        catalog_result=[model_info()],
        chat_responses=[chat_result_with_stats(wall_clock_ms=100.0)],
        residency_sequence=[[], resident(), resident(), resident()],
    )
    assert not hasattr(lms, "load_model")
    cfg = RunConfig(modelKey="qwen/qwen3-4b-2507", sessionId=None, referenceKey=None)
    run, _ = run_pack(_item_pack(), cfg, lmstudio=lms, root=tmp_path)
    assert run.fingerprint.get("residentModelsAtStart") == []


# ==================================================================================================
# 9. U2.1 — the fit manifest and AC-3's leakage refusal, in `_prepare_decision_arm` (impl review
#    Pass 6, P6-m2): every candidate, decision-only included, before any call
# ==================================================================================================


def _fixture_input_hash(item: dict[str, Any]) -> str:
    """v1.3 §3.5's `input_hash` for `guard-judge`, transcribed rather than imported."""
    fields = {k: item[k] for k in ("condition", "understanding", "turns", "path") if k in item}
    text = json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fixture_items() -> list[dict[str, Any]]:
    return [json.loads(line) for line in (_FIXTURE_PACK / "items.jsonl").read_text().splitlines()]


def _overlapping_workspace(root: Path, item_index: int, **kw: Any) -> None:
    """A manifest of the default calibration inputs plus one fixture item's input hash, declared
    with its real sha256 and size — a well-formed manifest that leaks exactly that item."""
    hashes = tuple(sorted((*_MANIFEST_HASHES, _fixture_input_hash(_fixture_items()[item_index]))))
    content = _manifest_bytes(hashes)
    manifest = {
        "file": f"{_CANDIDATE_ID}.fit.txt",
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(hashes),
        "purposes": ["question-tuning", "cutoff"],
    }
    _workspace(root, fit_manifest_bytes=content, fitManifest=manifest, **kw)


def test_a_decision_only_candidate_with_an_overlapping_manifest_refuses_with_exit_4(tmp_path):
    """P6-m2's case: `handoffModel: null`, so a hand-off-only check would have let it run."""
    _overlapping_workspace(tmp_path, 2)
    assert json.loads((tmp_path / "candidates" / f"{_CANDIDATE_ID}.json").read_text())[
        "handoffModel"
    ] is None
    exc = _refused(tmp_path, DecisionStub())
    assert exc.exitCode == 4 and "AC-3" in str(exc) and "fx-ca-01" in str(exc)
    assert "1 of this pack's items" in str(exc)


def test_an_overlap_with_a_side_check_item_refuses_too(tmp_path):
    """v1.3 §3.5: the manifest is intersected with eval **and** side-check items."""

    def side_check_last(row):
        if row["itemId"] == "fx-ca-02":
            row["stratum"] = "side-check"
        return row

    _overlapping_workspace(tmp_path, 3, items_edit=side_check_last)
    exc = _refused(tmp_path, DecisionStub())
    assert exc.exitCode == 4 and "fx-ca-02" in str(exc)


def test_a_manifest_whose_bytes_differ_from_the_declared_sha256_refuses_with_exit_4(tmp_path):
    """Same entry count as declared, different content: only the sha256 check can catch it."""
    other = tuple(sorted(hashlib.sha256(f"other-{n}".encode()).hexdigest() for n in (1, 2)))
    _workspace(tmp_path, fit_manifest_bytes=_manifest_bytes(other))
    exc = _refused(tmp_path, DecisionStub())
    assert exc.exitCode == 4 and "fitManifest.sha256" in str(exc)


def test_a_manifest_whose_entry_count_differs_from_the_declared_size_refuses_with_exit_4(
    tmp_path,
):
    """Right bytes, wrong declared size: only the size check can catch it."""
    body = _candidate_body()
    _workspace(tmp_path, fitManifest={**body["fitManifest"], "size": 3})
    exc = _refused(tmp_path, DecisionStub())
    assert exc.exitCode == 4 and "fitManifest.size" in str(exc)


def test_a_missing_manifest_file_refuses_with_exit_4(tmp_path):
    _workspace(tmp_path)
    (tmp_path / "candidates" / f"{_CANDIDATE_ID}.fit.txt").unlink()
    exc = _refused(tmp_path, DecisionStub())
    assert exc.exitCode == 4 and "cannot read fit manifest" in str(exc)


def test_the_leakage_and_manifest_refusals_come_before_the_load_step(tmp_path):
    """An exit-4 refusal costs nothing: no model is loaded on the shared GPU to then refuse."""
    _overlapping_workspace(tmp_path, 0)
    lms = _absent_at_start()
    with pytest.raises(RunRefused) as caught:
        _run(tmp_path, lms)
    assert caught.value.exitCode == 4 and lms.load_calls == [] and lms.order == []


def test_cli_run_with_an_overlapping_manifest_exits_4_and_writes_nothing(
    tmp_path, monkeypatch, capsys
):
    """On a contract-valid pack, so the exit 4 is the leakage refusal's, not `validate_pack`'s."""
    _overlapping_workspace(tmp_path, 1, contract_valid=True)
    stub = DecisionStub()
    monkeypatch.setattr(cli, "LMStudio", lambda base_url: stub)
    code = cli.main(
        ["run", "--root", str(tmp_path), "--pack", _PACK_ID, "--candidate", _CANDIDATE_ID]
    )
    assert code == 4 and stub.decide_calls == []
    assert "AC-3" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


# ==================================================================================================
# 10. P7-n1 (impl review Pass 7): a residency probe failing with `LMStudioCallFailed` — a non-200
#     GET or a malformed catalog body — at the load rule's probe and at step 10's end capture
# ==================================================================================================


def _probe_failure() -> LMStudioCallFailed:
    return LMStudioCallFailed("GET /api/v0/models: HTTP 500", status=500, body=None)


def test_a_load_rule_probe_that_fails_with_a_call_failure_records_assumed(tmp_path):
    """P7-n1's row: the probe raises `LMStudioCallFailed`, the run is `assumed` with no re-check
    call, and the run whose items all completed is kept rather than lost to a traceback."""
    _workspace(tmp_path)
    residency = [*[list(_LAYA_RESIDENT)] * 6, _probe_failure()]  # start, baseline, 4 items, rule
    lms = DecisionStub(decide_responses=_four(0.9, 0.1, 0.9, 0.9), residency_sequence=residency)
    run, _ = _run(tmp_path, lms)
    assert run.basis == "assumed" and len(lms.decide_calls) == 4
    store(run, tmp_path)
    valid, invalid = load_history(tmp_path, packId=_PACK_ID)
    assert invalid == [] and [r.basis for r in valid] == ["assumed"]


def _run_with_sleep(root: Path, lms: DecisionStub) -> tuple[Any, list[float]]:
    slept: list[float] = []
    pack = load_pack(root / "packs" / _PACK_ID)
    run, _ = run_pack(
        pack, _cfg(), lmstudio=lms, root=root, clock=_Clock(0.25), sleep=slept.append
    )
    return run, slept


def test_step_10_retries_a_call_failure_and_records_the_snapshot_that_answers(tmp_path):
    """Plan §3.4.4a step 10: a transient failure of the end-of-run GET is retried at 1 s, not
    allowed to escape — a non-200 included (the same `residency()` failure class as P7-n1)."""
    _workspace(tmp_path)
    residency = [*[list(_LAYA_RESIDENT)] * 7, _probe_failure(), list(_QWEN_RESIDENT)]
    lms = DecisionStub(decide_responses=_four(0.9, 0.1, 0.9, 0.9), residency_sequence=residency)
    run, slept = _run_with_sleep(tmp_path, lms)
    assert slept == [1.0]
    assert run.fingerprint.get("residentModelsAtEnd") == [
        {"id": "qwen/qwen3-4b-2507", "state": "loaded"}
    ]


def test_step_10_refuses_with_exit_5_after_three_retries_of_a_call_failure(tmp_path):
    _workspace(tmp_path)
    residency = [*[list(_LAYA_RESIDENT)] * 7, *[_probe_failure() for _ in range(4)]]
    lms = DecisionStub(decide_responses=_four(0.9, 0.1, 0.9, 0.9), residency_sequence=residency)
    with pytest.raises(RunRefused) as caught:
        _run_with_sleep(tmp_path, lms)
    assert caught.value.exitCode == 5 and "three retries" in str(caught.value)
