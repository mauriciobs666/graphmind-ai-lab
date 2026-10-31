"""`modelbench/decisions.py` — typed-decision-models U1c (plan v1.3 §3.3, §3.5, §3.6, §4 U1c;
plan2 v1.1 §3.2, §3.3, §4.1 U1c, §5).

The golden vectors under `tests/fixtures/decision-contract/` are model-bench's canonical copy
(v1.3 §3.2.2). Their expected outputs were derived **without** this module: the policy rows by
hand from §3.3's inequalities, the `input_hash` rows by hand-writing each canonical JSON string and
hashing its bytes with `sha256sum`.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any

import pytest

from modelbench import decisions, roles
from modelbench.decisions import (
    BAR_KIND_BY_ROLE,
    DECISION_POINT_CHAT_TIMEOUT_SECONDS,
    FIT_OUTCOMES,
    FR2_MIN_SPEEDUP,
    FR10_MEAN_LATENCY_MS_MAX,
    FR10_PER_ANSWER_AGREEMENT,
    HANDOFF_FAILURE_BY_ROLE,
    INPUT_FIELDS_BY_ROLE,
    MIN_EVAL_ITEMS,
    QUALITY_TARGETS_BY_BAR_KIND,
    Candidate,
    CandidateError,
    PolicyDecision,
    apply_policy,
    bar_holds,
    eval_set_hash,
    input_hash,
)
from modelbench.lmstudio import DecisionAnswer

VECTORS = Path(__file__).parent / "fixtures" / "decision-contract"


def _rows(name: str) -> list[dict[str, Any]]:
    with (VECTORS / name).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --- apply_policy against apply-policy.jsonl (v1.3 §3.3, §5 U1c) -------------------------------


def _answer(wire: dict[str, Any]) -> DecisionAnswer:
    """A vector row's §2.3 wire-shape answer as the `DecisionAnswer` `decide` returns."""
    return DecisionAnswer(
        name="q",
        type=wire["type"],
        probability=wire.get("probability"),
        choice=wire.get("choice"),
        probabilities=None,
        confidence=wire.get("confidence"),
    )


_POLICY_ROWS = _rows("apply-policy.jsonl")


@pytest.mark.parametrize("row", _POLICY_ROWS, ids=[r["case"] for r in _POLICY_ROWS])
def test_apply_policy_matches_the_golden_vector(row):
    got = apply_policy(row["policy"], _answer(row["answer"]))
    assert got == PolicyDecision(**row["expected"])


def test_the_policy_vectors_cover_the_boundaries_the_plan_names():
    """v1.3 §3.2.2: the file includes `p == l`, `p == h`, `l == h`, `confidence == c`, `c == 0`."""
    bands = [r for r in _POLICY_ROWS if r["policy"]["kind"] == "predicate-band"]
    choices = [r for r in _POLICY_ROWS if r["policy"]["kind"] == "choice-confidence"]
    p = lambda r: r["answer"]["probability"]  # noqa: E731
    assert any(p(r) == r["policy"]["noBelow"] < r["policy"]["yesAtOrAbove"] for r in bands)
    assert any(p(r) == r["policy"]["yesAtOrAbove"] for r in bands)
    assert any(r["policy"]["noBelow"] == r["policy"]["yesAtOrAbove"] for r in bands)
    assert any(r["answer"]["confidence"] == r["policy"]["minConfidence"] for r in choices)
    assert any(r["policy"]["minConfidence"] == 0 for r in choices)


def test_a_hand_off_carries_no_answer():
    decision = apply_policy(
        {"kind": "predicate-band", "yesAtOrAbove": 0.9, "noBelow": 0.1},
        _answer({"type": "predicate", "probability": 0.5}),
    )
    assert decision == PolicyDecision(handoff=True, answer=None)


@pytest.mark.parametrize(
    ("policy", "answer"),
    [
        (
            {"kind": "predicate-band", "yesAtOrAbove": 0.9, "noBelow": 0.1},
            {"type": "choice", "choice": "order", "confidence": 0.9},
        ),
        (
            {"kind": "choice-confidence", "minConfidence": 0.5},
            {"type": "predicate", "probability": 0.9},
        ),
        ({"kind": "per-label", "minConfidence": 0.5}, {"type": "predicate", "probability": 0.9}),
    ],
    ids=["band-over-a-choice", "confidence-over-a-predicate", "unknown-kind"],
)
def test_apply_policy_refuses_a_policy_that_does_not_fit_the_answer(policy, answer):
    with pytest.raises(ValueError):
        apply_policy(policy, _answer(answer))


# --- input_hash / INPUT_FIELDS_BY_ROLE against input-hash.jsonl (v1.3 §3.5, §5 U1c) -------------

_HASH_ROWS = _rows("input-hash.jsonl")


@pytest.mark.parametrize("row", _HASH_ROWS, ids=[r["case"] for r in _HASH_ROWS])
def test_input_hash_matches_the_golden_vector(row):
    assert input_hash(row["item"], row["role"]) == row["expected"]


def test_the_guard_hash_vectors_reach_every_input_field_and_a_non_input_one():
    """What makes a shrink **and** a widen of the `guard-judge` row each redden: every input field
    is present in some vector item (a dropped one changes that hash), and the items carry the
    non-input fields v1.3 §3.2.2 names (an added one changes the hash)."""
    items = [r["item"] for r in _HASH_ROWS if r["role"] == "guard-judge"]
    for field in INPUT_FIELDS_BY_ROLE["guard-judge"]:
        assert any(field in item for item in items), field
    for field in ("itemId", "tier", "expected", "stratum"):
        assert all(field in item for item in items), field


def test_input_fields_by_role_is_the_guard_row_only():
    """v1.3 §3.5's `guard-judge` row, transcribed; U4 adds the other two roles' rows."""
    assert dict(INPUT_FIELDS_BY_ROLE) == {
        "guard-judge": ("condition", "understanding", "turns", "path"),
    }


def test_input_hash_refuses_a_role_with_no_input_fields():
    with pytest.raises(ValueError, match="join-in-judge"):
        input_hash({"agentName": "a"}, "join-in-judge")


# --- eval_set_hash (v1.3 §3.5 FR-4) ---------------------------------------------------------------


def _hash_item(case_prefix: str) -> dict[str, Any]:
    (row,) = [r for r in _HASH_ROWS if r["case"].startswith(case_prefix)]
    return dict(row["item"])


#: Three `eval` items and one `side-check` item. The expected hash was computed with `sha256sum`
#: over the hand-built text: each `<input hash>:<json.dumps(expected)>` line, sorted, each ending
#: in `\n` — `876a…:false`, `876a…:true`, `a479…:false` (the side-check item is excluded).
_EVAL_SET_HASH = "3af02801e129e0486e1ceafff2b3617b74899cbf88c01a6614c5f271a627da7c"


def _eval_set() -> list[dict[str, Any]]:
    a = _hash_item("understanding path")
    b = _hash_item("turns path")
    a_flipped = {**_hash_item("understanding path"), "itemId": "gj-003", "expected": True}
    side = _hash_item("an absent input field")
    return [b, side, a_flipped, a]


def test_eval_set_hash_is_the_sorted_input_hash_and_label_lines_over_the_eval_stratum():
    assert eval_set_hash(_eval_set(), "guard-judge") == _EVAL_SET_HASH


def test_eval_set_hash_ignores_item_order_and_the_side_check_stratum():
    items = _eval_set()
    items.reverse()
    without_side = [i for i in items if i["stratum"] == "eval"]
    assert eval_set_hash(items, "guard-judge") == _EVAL_SET_HASH
    assert eval_set_hash(without_side, "guard-judge") == _EVAL_SET_HASH


def test_eval_set_hash_changes_with_a_label_but_not_with_an_item_id():
    items = _eval_set()
    relabeled = [dict(i) for i in items]
    relabeled[0]["expected"] = not relabeled[0]["expected"]
    renamed = [{**i, "itemId": i["itemId"] + "-v2"} for i in items]
    assert eval_set_hash(relabeled, "guard-judge") != _EVAL_SET_HASH
    assert eval_set_hash(renamed, "guard-judge") == _EVAL_SET_HASH


def test_eval_set_hash_refuses_an_item_with_no_stratum():
    """U3b makes `stratum` required with no default; the hash never assumes one either."""
    items = _eval_set()
    del items[0]["stratum"]
    with pytest.raises(ValueError, match="stratum"):
        eval_set_hash(items, "guard-judge")


# --- requirement constants and bar_holds (v1.3 §3.5, §3.6; §5 U1c) ------------------------------


def test_the_requirement_constants_are_the_requirements_values():
    """Each a requirement value (FR-11, FR-2, FR-10), never pack data. Transcribed from
    `docs/requirements/typed-decision-models.md` via plan v1.3 §3.5."""
    assert MIN_EVAL_ITEMS == 200
    assert FR2_MIN_SPEEDUP == 2.0
    assert FR10_PER_ANSWER_AGREEMENT == 0.90
    assert FR10_MEAN_LATENCY_MS_MAX == 350


def test_the_decision_point_chat_timeout_is_180_seconds():
    """Plan2 §3.2 (note §Q4.3): the one chat timeout a decision-point run may carry a verdict at."""
    assert DECISION_POINT_CHAT_TIMEOUT_SECONDS == 180


@pytest.mark.parametrize(
    ("measured", "threshold", "direction", "met"),
    [
        (350, FR10_MEAN_LATENCY_MS_MAX, "at-most", True),
        (351, FR10_MEAN_LATENCY_MS_MAX, "at-most", False),
        (0.90, FR10_PER_ANSWER_AGREEMENT, "at-least", True),
        (0.899, FR10_PER_ANSWER_AGREEMENT, "at-least", False),
        (2.0, FR2_MIN_SPEEDUP, "at-least", True),
        (1.999, FR2_MIN_SPEEDUP, "at-least", False),
        (200, MIN_EVAL_ITEMS, "at-least", True),
        (199, MIN_EVAL_ITEMS, "at-least", False),
    ],
)
def test_bar_holds_on_the_measured_value_against_each_requirement_constant(
    measured, threshold, direction, met
):
    """§5 U1c. Measured values are literals, so mutating any constant either way moves one of a
    pair across its boundary."""
    assert bar_holds(measured, threshold, direction=direction) is met


def test_bar_holds_takes_no_interval_and_direction_has_no_default():
    """FR-11a: the bar is judged on the measured value; the interval is printed, never passed."""
    params = inspect.signature(bar_holds).parameters
    assert list(params) == ["measured", "threshold", "direction"]
    assert params["direction"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["direction"].default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        bar_holds(1.0, 1.0)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("measured", "threshold", "direction"),
    [
        (1.0, 1.0, "at_least"),
        (1.0, 1.0, "above"),
        (math.nan, 1.0, "at-least"),
        (math.inf, 1.0, "at-most"),
        (1.0, math.nan, "at-most"),
        (True, 1.0, "at-least"),
    ],
)
def test_bar_holds_refuses_an_unknown_direction_or_a_non_finite_value(
    measured, threshold, direction
):
    with pytest.raises(ValueError):
        bar_holds(measured, threshold, direction=direction)


# --- the pinned tables (v1.3 §3.5; plan2 §3.2, §3.3) ---------------------------------------------


def test_bar_kind_by_role_is_the_guard_row_only():
    """v1.3 §3.5, transcribed — `guard-judge` is the speed-up point; U4 adds the two new-ability
    roles. A pack cannot pick its own bar."""
    assert dict(BAR_KIND_BY_ROLE) == {"guard-judge": "speed-up"}
    assert set(BAR_KIND_BY_ROLE) <= set(roles.ROLES)
    assert set(BAR_KIND_BY_ROLE.values()) <= set(QUALITY_TARGETS_BY_BAR_KIND)


def test_handoff_failure_by_role_has_exactly_the_bar_kind_roles():
    """Plan2 §3.3: U1c pins `HANDOFF_FAILURE_BY_ROLE`'s keys to `BAR_KIND_BY_ROLE`'s."""
    assert set(HANDOFF_FAILURE_BY_ROLE) == set(BAR_KIND_BY_ROLE)


def test_handoff_failure_by_role_is_the_guard_row_only():
    """Plan2 §3.3's `guard-judge` row, transcribed (U2's withheld-versus-kept test is the
    behavioural value pin; this is the third mutation, a moved value, until then)."""
    assert dict(HANDOFF_FAILURE_BY_ROLE) == {"guard-judge": "withhold"}


def test_fit_outcomes_and_quality_targets_are_the_notes_sets():
    """Plan2 §3.2 / note §Q2.0–§Q2.2, transcribed."""
    assert FIT_OUTCOMES == frozenset(
        {
            "within-margin",
            "below-margin",
            "fit-infeasible: ranking",
            "fit-infeasible: agreement",
            "fit-infeasible: latency",
        }
    )
    assert {k: set(v) for k, v in QUALITY_TARGETS_BY_BAR_KIND.items()} == {
        "speed-up": {"D_c<=0", "mcnemar-nonreject"},
        "new-ability": {"T=0.94", "T=0.93", "T=0.92", "T=0.91", "T=0.90", "max-min-agreement"},
    }


def test_the_tables_are_read_only():
    for table in (BAR_KIND_BY_ROLE, HANDOFF_FAILURE_BY_ROLE, INPUT_FIELDS_BY_ROLE,
                  QUALITY_TARGETS_BY_BAR_KIND):
        with pytest.raises(TypeError):
            table["x"] = "y"  # type: ignore[index]


# --- Candidate.load (v1.3 §3.3; plan2 §3.2, §5 U1c) ----------------------------------------------


def _valid() -> dict[str, Any]:
    return {
        "candidateId": "guard-laya-band-v1",
        "role": "guard-judge",
        "decisionModel": "laya",
        "question": {
            "type": "predicate",
            "name": "advance",
            "instructions": "Is the condition satisfied?",
        },
        "policy": {"kind": "predicate-band", "yesAtOrAbove": 0.92, "noBelow": 0.08},
        "handoffModel": "qwen/qwen3-4b-2507",
        "decisionTimeoutSeconds": 2.0,
        "fit": {"outcome": "below-margin", "qualityTarget": "mcnemar-nonreject"},
        "fitManifest": {
            "file": "guard-laya-band-v1.fit.txt",
            "sha256": "0" * 64,
            "size": 1400,
            "purposes": ["question-tuning", "cutoff"],
        },
        "provenance": {
            "builtBy": "decision-models",
            "datasetVersion": "2026-10-30",
            "sourceSha": "825d5d93",
        },
    }


def _write(root: Path, body: Any, *, candidate_id: str = "guard-laya-band-v1") -> Path:
    directory = root / "candidates"
    directory.mkdir(exist_ok=True)
    path = directory / f"{candidate_id}.json"
    path.write_text(body if isinstance(body, str) else json.dumps(body, indent=2), "utf-8")
    return path


def _load(root: Path, body: Any) -> Candidate:
    _write(root, body)
    return Candidate.load(root, "guard-laya-band-v1")


def _refused(root: Path, body: Any, match: str | None = None) -> CandidateError:
    with pytest.raises(CandidateError, match=match) as caught:
        _load(root, body)
    return caught.value


def _with(path: tuple[str, ...], value: Any) -> dict[str, Any]:
    body = _valid()
    target = body
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return body


def _without(path: tuple[str, ...]) -> dict[str, Any]:
    body = _valid()
    target = body
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]
    return body


def test_candidate_load_reads_candidates_dir_and_keeps_every_field(tmp_path):
    path = _write(tmp_path, _valid())
    c = Candidate.load(tmp_path, "guard-laya-band-v1")
    assert c.candidateId == "guard-laya-band-v1"
    assert c.role == "guard-judge"
    assert c.decisionModel == "laya"
    assert c.question == _valid()["question"]
    assert c.policy == _valid()["policy"]
    assert c.handoffModel == "qwen/qwen3-4b-2507"
    assert c.decisionTimeoutSeconds == 2.0
    assert c.fit == _valid()["fit"]
    assert c.fitManifest == _valid()["fitManifest"]
    assert c.provenance == _valid()["provenance"]
    assert c.contentHash == hashlib.sha256(path.read_bytes()).hexdigest()


def test_candidate_content_hash_changes_on_any_in_place_edit(tmp_path):
    """v1.3 §3.5: an in-place edit under the same `candidateId` is a new candidate for K."""
    before = _load(tmp_path, _valid()).contentHash
    after = _load(tmp_path, json.dumps(_valid(), indent=4)).contentHash
    assert before != after


def test_candidate_load_accepts_an_integer_timeout_as_a_float(tmp_path):
    c = _load(tmp_path, _with(("decisionTimeoutSeconds",), 3))
    assert c.decisionTimeoutSeconds == 3.0 and isinstance(c.decisionTimeoutSeconds, float)


def test_candidate_load_refuses_a_score_question_up_front(tmp_path):
    """Plan2 §3.2 (impl review U1a-n1): `decide` refuses a `score` answer only after the POST, so
    without this every item would become `unrunnable`."""
    body = _with(("question",), {"type": "score", "name": "urgency", "instructions": "How?"})
    # ...refused **as** a score question: the message is the operator's only clue why.
    _refused(tmp_path, body, match="question.type 'score' is refused: no probe")


@pytest.mark.parametrize(
    "value",
    [0, 0.0, -1.0, math.nan, math.inf, -math.inf, True, False, "2.0", None],
    ids=["0", "0.0", "negative", "nan", "inf", "-inf", "true", "false", "string", "null"],
)
def test_candidate_load_refuses_an_ill_typed_decision_timeout(tmp_path, value):
    """`json.dumps` writes the bare `NaN`/`Infinity` tokens, which `json.loads` reads back."""
    _refused(tmp_path, _with(("decisionTimeoutSeconds",), value), match="decisionTimeoutSeconds")


@pytest.mark.parametrize(
    "key",
    [
        "candidateId",
        "role",
        "decisionModel",
        "question",
        "policy",
        "handoffModel",
        "decisionTimeoutSeconds",
        "fit",
        "fitManifest",
        "provenance",
    ],
)
def test_candidate_load_refuses_a_missing_top_level_field(tmp_path, key):
    """Every field is required — `handoffModel` too, as an explicit `null` when there is none."""
    _refused(tmp_path, _without((key,)), match=key)


@pytest.mark.parametrize("outcome", sorted(FIT_OUTCOMES))
def test_candidate_load_accepts_every_fit_outcome(tmp_path, outcome):
    """FR-4: every candidate is evaluated whatever its outcome, so none is refused."""
    assert _load(tmp_path, _with(("fit", "outcome"), outcome)).fit["outcome"] == outcome


@pytest.mark.parametrize(
    "outcome", ["fit-infeasible", "fit-infeasible: speed", "Within-margin", "", None, 1]
)
def test_candidate_load_refuses_an_unknown_fit_outcome(tmp_path, outcome):
    _refused(tmp_path, _with(("fit", "outcome"), outcome), match="fit.outcome")


@pytest.mark.parametrize("target", ["D_c<=0", "mcnemar-nonreject"])
def test_candidate_load_accepts_each_speed_up_quality_target_on_the_guard(tmp_path, target):
    assert _load(tmp_path, _with(("fit", "qualityTarget"), target)).fit["qualityTarget"] == target


@pytest.mark.parametrize(
    "target", ["T=0.94", "T=0.93", "T=0.92", "T=0.91", "T=0.90", "max-min-agreement"]
)
def test_candidate_load_refuses_a_new_ability_quality_target_on_the_guard(tmp_path, target):
    """Plan2 §3.2: `fit.qualityTarget` is closed per bar kind, through `BAR_KIND_BY_ROLE`."""
    _refused(tmp_path, _with(("fit", "qualityTarget"), target), match="qualityTarget")


@pytest.mark.parametrize("target", ["T=0.95", "D_c<0", "", None])
def test_candidate_load_refuses_an_unknown_quality_target(tmp_path, target):
    _refused(tmp_path, _with(("fit", "qualityTarget"), target), match="qualityTarget")


@pytest.mark.parametrize(
    "fit",
    [None, "below-margin", [], {"outcome": "below-margin"}, {"qualityTarget": "D_c<=0"}],
    ids=["null", "string", "list", "no-qualityTarget", "no-outcome"],
)
def test_candidate_load_refuses_an_ill_shaped_fit(tmp_path, fit):
    _refused(tmp_path, _with(("fit",), fit), match="fit")


@pytest.mark.parametrize("role", ["join-in-judge", "embedder", "no-such-role"])
def test_candidate_load_refuses_a_role_with_no_bar_kind(tmp_path, role):
    """Only `BAR_KIND_BY_ROLE`'s roles are decision points (the other two arrive with U4)."""
    _refused(tmp_path, _with(("role",), role), match="role")


def test_candidate_load_refuses_a_file_whose_id_disagrees_with_its_name(tmp_path):
    _refused(tmp_path, _with(("candidateId",), "guard-laya-band-v2"), match="candidateId")


@pytest.mark.parametrize("candidate_id", ["../escape", "a/b", "", ".", ".."])
def test_candidate_load_refuses_an_id_that_is_not_a_plain_file_name(tmp_path, candidate_id):
    with pytest.raises(CandidateError, match="candidateId"):
        Candidate.load(tmp_path, candidate_id)


def test_candidate_load_refuses_a_missing_file_naming_the_path(tmp_path):
    with pytest.raises(CandidateError, match="no-such-candidate"):
        Candidate.load(tmp_path, "no-such-candidate")


@pytest.mark.parametrize("raw", ["{not json", "[]", '"a string"'])
def test_candidate_load_refuses_a_body_that_is_not_a_json_object(tmp_path, raw):
    _refused(tmp_path, raw)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("decisionModel",), ""),
        (("decisionModel",), 7),
        (("handoffModel",), ""),
        (("handoffModel",), 7),
        (("role",), None),
        (("question",), "advance"),
        (("question", "name"), ""),
        (("question", "instructions"), None),
        (("policy",), None),
        (("fitManifest",), None),
        (("fitManifest", "file"), ""),
        (("fitManifest", "sha256"), "abc"),
        (("fitManifest", "sha256"), "G" * 64),
        (("fitManifest", "size"), -1),
        (("fitManifest", "size"), 1.5),
        (("fitManifest", "size"), True),
        (("fitManifest", "purposes"), "cutoff"),
        (("fitManifest", "purposes"), [1]),
        (("provenance",), None),
        (("provenance", "builtBy"), ""),
    ],
    ids=lambda v: ".".join(v) if isinstance(v, tuple) else repr(v),
)
def test_candidate_load_refuses_an_ill_typed_field(tmp_path, path, value):
    _refused(tmp_path, _with(path, value), match=path[0])


@pytest.mark.parametrize("key", ["file", "sha256", "size", "purposes"])
def test_candidate_load_refuses_a_fit_manifest_missing_a_key(tmp_path, key):
    _refused(tmp_path, _without(("fitManifest", key)), match="fitManifest")


@pytest.mark.parametrize("key", ["builtBy", "datasetVersion", "sourceSha"])
def test_candidate_load_refuses_a_provenance_missing_a_key(tmp_path, key):
    _refused(tmp_path, _without(("provenance", key)), match="provenance")


# --- policy validation, through Candidate.load (v1.3 §3.3; §4 U1c) -------------------------------


def _band(low: Any, high: Any, *, handoff: str | None = "qwen/qwen3-4b-2507") -> dict[str, Any]:
    body = _with(("policy",), {"kind": "predicate-band", "yesAtOrAbove": high, "noBelow": low})
    body["handoffModel"] = handoff
    return body


def _choice_candidate(c: Any, *, handoff: str | None = "qwen/qwen3-4b-2507") -> dict[str, Any]:
    body = _valid()
    body["question"] = {
        "type": "choice",
        "name": "messageType",
        "instructions": "What kind of message is this?",
        "choices": [
            {"value": "order", "instructions": "places an order"},
            {"value": "other", "instructions": "anything else"},
        ],
    }
    body["policy"] = {"kind": "choice-confidence", "minConfidence": c}
    body["handoffModel"] = handoff
    return body


@pytest.mark.parametrize(
    ("low", "high"), [(0.0, 1.0), (0.08, 0.92), (0.0, 0.5), (0.5, 1.0), (0.3, 0.30000001)]
)
def test_candidate_load_accepts_a_band_with_a_hand_off_model(tmp_path, low, high):
    assert _load(tmp_path, _band(low, high)).policy["noBelow"] == low


@pytest.mark.parametrize("edge", [0.0, 0.5, 1.0])
def test_candidate_load_accepts_an_empty_band_with_no_hand_off_model(tmp_path, edge):
    assert _load(tmp_path, _band(edge, edge, handoff=None)).handoffModel is None


def test_candidate_load_refuses_an_empty_band_with_a_hand_off_model(tmp_path):
    """Empty band iff no hand-off: a hand-off model the policy can never reach is a mistake."""
    _refused(tmp_path, _band(0.5, 0.5), match="hand")


def test_candidate_load_refuses_a_band_with_no_hand_off_model(tmp_path):
    """v1.3 §3.3: `l == h` is required when `handoffModel` is null."""
    _refused(tmp_path, _band(0.08, 0.92, handoff=None), match="hand")


@pytest.mark.parametrize(
    ("low", "high"),
    [
        (0.92, 0.08),
        (-0.01, 0.5),
        (0.5, 1.01),
        (math.nan, 0.5),
        (0.1, math.inf),
        (True, 0.5),
        (0.1, "0.9"),
        (None, 0.5),
    ],
)
def test_candidate_load_refuses_a_band_outside_zero_le_l_le_h_le_one(tmp_path, low, high):
    _refused(tmp_path, _band(low, high), match="policy")


@pytest.mark.parametrize("c", [0.0001, 0.5, 1.0])
def test_candidate_load_accepts_a_confidence_policy_with_a_hand_off_model(tmp_path, c):
    assert _load(tmp_path, _choice_candidate(c)).policy["minConfidence"] == c


def test_candidate_load_accepts_c_zero_with_no_hand_off_model(tmp_path):
    assert _load(tmp_path, _choice_candidate(0.0, handoff=None)).handoffModel is None


def test_candidate_load_refuses_c_zero_with_a_hand_off_model(tmp_path):
    _refused(tmp_path, _choice_candidate(0.0), match="hand")


def test_candidate_load_refuses_a_positive_c_with_no_hand_off_model(tmp_path):
    _refused(tmp_path, _choice_candidate(0.5, handoff=None), match="hand")


@pytest.mark.parametrize("c", [-0.1, 1.1, math.nan, True, "0.5", None])
def test_candidate_load_refuses_a_confidence_outside_zero_one(tmp_path, c):
    _refused(tmp_path, _choice_candidate(c), match="policy")


def test_candidate_load_refuses_a_policy_kind_that_does_not_fit_the_question(tmp_path):
    body = _choice_candidate(0.5)
    body["policy"] = {"kind": "predicate-band", "yesAtOrAbove": 0.9, "noBelow": 0.1}
    _refused(tmp_path, body, match="policy")
    _refused(
        tmp_path, _with(("policy",), {"kind": "choice-confidence", "minConfidence": 0.5}),
        match="policy",
    )


def test_candidate_load_refuses_an_unknown_policy_kind(tmp_path):
    _refused(tmp_path, _with(("policy", "kind"), "per-label"), match="policy")


@pytest.mark.parametrize(
    "choices",
    [
        None,
        [],
        ["order", "other"],
        [{"value": "order", "instructions": "x"}, {"value": "order", "instructions": "y"}],
        [{"value": "", "instructions": "x"}],
        [{"instructions": "x"}],
    ],
    ids=["missing", "empty", "bare-strings", "duplicate-values", "empty-value", "no-value"],
)
def test_candidate_load_refuses_ill_formed_choices(tmp_path, choices):
    """§2.3: choice options are `{value, instructions}` objects (bare strings are a server 400)."""
    body = _choice_candidate(0.5)
    if choices is None:
        del body["question"]["choices"]
    else:
        body["question"]["choices"] = choices
    _refused(tmp_path, body, match="choices")


def test_candidate_load_refuses_choices_on_a_predicate_question(tmp_path):
    body = _with(("question", "choices"), [{"value": "yes", "instructions": "x"}])
    _refused(tmp_path, body, match="choices")


def test_candidate_error_is_a_value_error():
    assert issubclass(CandidateError, ValueError)


def test_decisions_is_pure_stdlib():
    """FR-23 / zero runtime dependencies: nothing outside the stdlib and `modelbench`."""
    import ast
    import sys

    tree = ast.parse(Path(decisions.__file__).read_text("utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""] if node.level == 0 else ["modelbench"]
        else:
            continue
        for name in names:
            top = name.split(".")[0]
            assert top in sys.stdlib_module_names or top == "modelbench", name


@pytest.mark.parametrize(
    "path",
    [("role",), ("question", "type"), ("policy", "kind"), ("fit", "outcome"),
     ("fit", "qualityTarget")],
    ids=lambda p: ".".join(p),
)
@pytest.mark.parametrize("value", [[], {}], ids=["list", "object"])
def test_candidate_load_refuses_an_unhashable_value_as_a_candidate_error(tmp_path, path, value):
    """A JSON list or object where a closed-set string belongs is a refusal, never a `TypeError`
    escaping a membership test."""
    _refused(tmp_path, _with(path, value), match=path[0])
