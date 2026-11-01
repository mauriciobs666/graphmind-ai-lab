"""`modelbench/decisions.py` — typed-decision-models U1c (plan v1.3 §3.3, §3.5, §3.6, §4 U1c;
plan2 v1.1 §3.2, §3.3, §4.1 U1c, §5; plan3 v1.2 §3.2, §3.3, §5 U1d).

The golden vectors under `tests/fixtures/decision-contract/` are model-bench's canonical copy
(v1.3 §3.2.2). Their expected outputs were derived **without** this module: the policy rows by
hand from §3.3's inequalities, the `input_hash` rows by hand-writing each canonical JSON string and
hashing its bytes with `sha256sum` (the `question_spec_hash` vectors below the same way), the
`over-limit.jsonl` rows by hand from plan3 §3.3's rule
(status `500` **and** the case-sensitive marker in the body), the `fit-manifest.jsonl` rows by
hand from the format its `_format` header row states, each `sha256` from `printf … | sha256sum`.
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
    ANSWER_SPACE_BY_ROLE,
    BAR_KIND_BY_ROLE,
    DECISION_POINT_CHAT_TIMEOUT_SECONDS,
    FIT_OUTCOMES,
    FR2_MIN_SPEEDUP,
    FR10_MEAN_LATENCY_MS_MAX,
    FR10_PER_ANSWER_AGREEMENT,
    HANDOFF_FAILURE_BY_ROLE,
    INPUT_FIELDS_BY_ROLE,
    MESSAGE_TYPE_LABELS,
    MIN_EVAL_ITEMS,
    OVER_LIMIT_MARKER,
    QUALITY_TARGETS_BY_BAR_KIND,
    SCORER_BY_ROLE,
    Candidate,
    CandidateError,
    PolicyDecision,
    apply_policy,
    bar_holds,
    eval_set_hash,
    input_hash,
    is_over_limit,
    question_spec_hash,
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


@pytest.mark.parametrize(
    "policy",
    [
        {"kind": "predicate-band", "yesAtOrAbove": 0.4, "noBelow": 0.6},
        {"kind": "predicate-band", "yesAtOrAbove": 0.9, "noBelow": -0.1},
        {"kind": "predicate-band", "yesAtOrAbove": 1.1, "noBelow": 0.1},
        {"kind": "predicate-band", "yesAtOrAbove": math.nan, "noBelow": 0.1},
        {"kind": "predicate-band", "yesAtOrAbove": 0.9, "noBelow": True},
        {"kind": "predicate-band", "yesAtOrAbove": 0.9},
    ],
    ids=["l-above-h", "l-below-0", "h-above-1", "h-nan", "l-bool", "l-missing"],
)
def test_apply_policy_refuses_a_band_outside_zero_le_l_le_h_le_one(policy):
    """Impl review Pass 2, P2-m2: `apply_policy` does not rely on `Candidate.load` having
    validated its band. Three copies exist (model-bench, falkor-chat U9, decision-models' fit
    search); on `l > h` the yes-first and no-first branch orders disagree (`p = 0.5` above is
    *yes* one way, *no* the other), so the band is refused before either branch runs."""
    with pytest.raises(ValueError, match="band"):
        apply_policy(policy, _answer({"type": "predicate", "probability": 0.5}))


@pytest.mark.parametrize("c", [-0.1, 1.1, math.nan, True, None], ids=repr)
def test_apply_policy_refuses_a_confidence_floor_outside_zero_one(c):
    policy = {"kind": "choice-confidence", "minConfidence": c}
    if c is None:
        del policy["minConfidence"]
    with pytest.raises(ValueError, match="minConfidence"):
        apply_policy(policy, _answer({"type": "choice", "choice": "order", "confidence": 0.9}))


@pytest.mark.parametrize(
    ("handoff", "answer"),
    [(False, None), (True, True), (True, False), (True, "order")],
    ids=["decided-with-no-answer", "handoff-with-yes", "handoff-with-no", "handoff-with-choice"],
)
def test_policy_decision_refuses_a_state_that_cannot_exist(handoff, answer):
    """Impl review Pass 2, P2-n2: `answer is None` iff `handoff` — a rule the copies transcribe."""
    with pytest.raises(ValueError, match="handoff"):
        PolicyDecision(handoff=handoff, answer=answer)


# --- is_over_limit against over-limit.jsonl (plan3 §3.3, §5 U1d) -------------------------------

_OVER_LIMIT_ROWS = _rows("over-limit.jsonl")
#: The `ws:agent-team` body document's (a1d388f4…) LM Studio 500 body: 232 bytes, verbatim.
_BODY_DOCUMENT_BYTES = 232


@pytest.mark.parametrize("row", _OVER_LIMIT_ROWS, ids=[r["case"] for r in _OVER_LIMIT_ROWS])
def test_is_over_limit_matches_the_golden_vector(row):
    assert is_over_limit(row["status"], row["body"]) is row["overLimit"]


def test_the_over_limit_vectors_cover_the_rows_the_plan_names():
    """Plan3 §3.3's list: the body document's body at 500 (true) and 400 (false), its marker
    upper-cased at 500 (false), another 500 body, the `/api/v0` HTTP-200 body, `""` and `null`.
    Without these rows a dropped status check or a case-folded body stays green."""
    by_case = {r["case"]: r for r in _OVER_LIMIT_ROWS}
    real = by_case["body-document-at-500"]
    assert len(real["body"].encode("utf-8")) == _BODY_DOCUMENT_BYTES
    assert (real["status"], real["overLimit"]) == (500, True)
    at_400 = by_case["body-document-at-400"]
    assert (at_400["body"], at_400["status"], at_400["overLimit"]) == (real["body"], 400, False)
    upper = by_case["marker-upper-cased-at-500"]
    assert upper["body"] == real["body"].replace("too large to process", "TOO LARGE TO PROCESS")
    assert upper["body"] != real["body"] and (upper["status"], upper["overLimit"]) == (500, False)
    assert {"another-500-body", "api-v0-http-200-error-body", "empty-body-at-500",
            "null-body-at-500"} <= set(by_case)
    assert all(r["overLimit"] is False for c, r in by_case.items() if c != "body-document-at-500")


def test_the_over_limit_marker_is_the_plans_text():
    """Plan3 §3.3, R-16: one LM Studio version's wording, pinned here and in each copy."""
    assert OVER_LIMIT_MARKER == "too large to process"


# --- input_hash / INPUT_FIELDS_BY_ROLE against input-hash.jsonl (v1.3 §3.5, §5 U1c) -------------

_HASH_ROWS = _rows("input-hash.jsonl")


@pytest.mark.parametrize("row", _HASH_ROWS, ids=[r["case"] for r in _HASH_ROWS])
def test_input_hash_matches_the_golden_vector(row):
    assert input_hash(row["item"], row["role"]) == row["expected"]


#: The non-input fields every vector item of a role carries (v1.3 §3.2.2; plan2 §3.5 adds `seedId`,
#: `labelVotes` and, on message type, `threadPosition` — none of them in `INPUT_FIELDS_BY_ROLE`).
_NON_INPUT_FIELDS_BY_ROLE = {
    "guard-judge": ("itemId", "tier", "expected", "stratum"),
    "join-in-judge": ("itemId", "expected", "stratum", "seedId", "labelVotes"),
    "message-type-classifier": (
        "itemId", "expected", "stratum", "seedId", "labelVotes", "threadPosition",
    ),
}


@pytest.mark.parametrize("role", sorted(_NON_INPUT_FIELDS_BY_ROLE))
def test_the_hash_vectors_reach_every_input_field_and_the_non_input_ones(role):
    """What makes a shrink **and** a widen of each role's row redden: every input field is present
    in some vector item of that role (a dropped one changes that hash), and the items carry the
    non-input fields (an added one changes the hash)."""
    items = [r["item"] for r in _HASH_ROWS if r["role"] == role]
    assert items, role
    for field in INPUT_FIELDS_BY_ROLE[role]:
        assert any(field in item for item in items), field
    for field in _NON_INPUT_FIELDS_BY_ROLE[role]:
        assert all(field in item for item in items), field


@pytest.mark.parametrize("role", ["join-in-judge", "message-type-classifier"])
def test_every_new_ability_hash_vector_is_in_the_canonical_form(role):
    """Coordinator ruling (U4.1): a new-ability item carries every input field, an empty one
    written, never omitted — `input_hash` omits an absent field, so the two forms would hash as
    two items. The opening join-in row pins `priorTurns: []`."""
    items = [r["item"] for r in _HASH_ROWS if r["role"] == role]
    for item in items:
        assert all(field in item for field in INPUT_FIELDS_BY_ROLE[role]), item["itemId"]
    if role == "join-in-judge":
        assert any(item["priorTurns"] == [] for item in items)
    else:
        assert any(item["priorTurn"] is None for item in items)


def test_every_input_fields_row_has_hash_vectors():
    assert {r["role"] for r in _HASH_ROWS} == set(INPUT_FIELDS_BY_ROLE)


def test_input_fields_by_role_is_the_plans_table():
    """v1.3 §3.5's rows, with plan2 §3.1's `message-type-classifier` row (`priorTurn`, note
    §Q1.5), transcribed — the move pin: a field moved to another role's row reddens here."""
    assert dict(INPUT_FIELDS_BY_ROLE) == {
        "guard-judge": ("condition", "understanding", "turns", "path"),
        "join-in-judge": ("agentName", "priorTurns", "newMessage"),
        "message-type-classifier": ("text", "priorTurn"),
    }


def test_input_hash_refuses_a_role_with_no_input_fields():
    with pytest.raises(ValueError, match="embedder"):
        input_hash({"text": "a"}, "embedder")


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


def test_bar_kind_by_role_is_the_plans_table():
    """v1.3 §3.5, transcribed — `guard-judge` is the speed-up point, `join-in-judge` and
    `message-type-classifier` the new abilities. A pack cannot pick its own bar."""
    assert dict(BAR_KIND_BY_ROLE) == {
        "guard-judge": "speed-up",
        "join-in-judge": "new-ability",
        "message-type-classifier": "new-ability",
    }
    assert set(BAR_KIND_BY_ROLE.values()) <= set(QUALITY_TARGETS_BY_BAR_KIND)


def test_the_decision_point_roles_are_roles_last_two_plus_the_guard():
    """Binds `BAR_KIND_BY_ROLE`'s keys to `roles.ROLES`: every decision point is a role, and the
    roles v1.3 §3.6 added for typed-decision-models are exactly its new abilities."""
    assert set(BAR_KIND_BY_ROLE) <= set(roles.ROLES)
    assert set(roles.ROLES[-2:]) == {r for r, k in BAR_KIND_BY_ROLE.items() if k == "new-ability"}


@pytest.mark.parametrize(
    "table", [HANDOFF_FAILURE_BY_ROLE, INPUT_FIELDS_BY_ROLE, ANSWER_SPACE_BY_ROLE, SCORER_BY_ROLE],
    ids=["HANDOFF_FAILURE_BY_ROLE", "INPUT_FIELDS_BY_ROLE", "ANSWER_SPACE_BY_ROLE",
         "SCORER_BY_ROLE"],
)
def test_each_decision_point_table_has_exactly_the_bar_kind_roles(table):
    """Plan2 §3.3 (U1c pins `HANDOFF_FAILURE_BY_ROLE`'s keys to `BAR_KIND_BY_ROLE`'s); the same
    binding for the input-field and answer-space tables."""
    assert set(table) == set(BAR_KIND_BY_ROLE)


def test_handoff_failure_by_role_is_the_plans_table():
    """Plan2 §3.3's three rows, transcribed: the guard withholds, the new abilities fail closed
    (U2's withheld-versus-kept tests are the guard's behavioural value pin)."""
    assert dict(HANDOFF_FAILURE_BY_ROLE) == {
        "guard-judge": "withhold",
        "join-in-judge": "fail-closed",
        "message-type-classifier": "fail-closed",
    }


def test_scorer_by_role_is_the_plans_table():
    """v1.3 §3.6, transcribed: the guard keeps `classification`, the new abilities score with
    `agreement` (impl review P22-m1; `validate_pack` refuses any other)."""
    assert dict(SCORER_BY_ROLE) == {
        "guard-judge": "classification",
        "join-in-judge": "agreement",
        "message-type-classifier": "agreement",
    }


def test_message_type_labels_are_fr10s_five():
    assert MESSAGE_TYPE_LABELS == ("order", "question", "complaint", "escalation", "other")


def test_answer_space_by_role_is_the_plans_answer_space_column():
    """v1.3 §3.2's table: the guard and join-in are predicates, message type a choice over
    FR-10's five labels."""
    assert {role: dict(space) for role, space in ANSWER_SPACE_BY_ROLE.items()} == {
        "guard-judge": {"type": "predicate"},
        "join-in-judge": {"type": "predicate"},
        "message-type-classifier": {
            "type": "choice", "labels": ("order", "question", "complaint", "escalation", "other"),
        },
    }


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
                  QUALITY_TARGETS_BY_BAR_KIND, ANSWER_SPACE_BY_ROLE, SCORER_BY_ROLE,
                  *ANSWER_SPACE_BY_ROLE.values()):
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


@pytest.mark.parametrize("role", ["chat-responder", "embedder", "no-such-role"])
def test_candidate_load_refuses_a_role_with_no_bar_kind(tmp_path, role):
    """Only `BAR_KIND_BY_ROLE`'s roles are decision points."""
    _refused(tmp_path, _with(("role",), role), match="role")


@pytest.mark.parametrize("role", ["join-in-judge", "message-type-classifier"])
def test_candidate_load_holds_a_new_ability_role_to_the_new_ability_targets(tmp_path, role):
    """The new-ability rows of `BAR_KIND_BY_ROLE`, behaviourally: a new-ability candidate loads
    with a new-ability quality target and is refused a speed-up one (plan2 §3.2)."""
    body = _valid()
    body["role"] = role
    if role == "message-type-classifier":
        body["question"] = {
            "type": "choice", "name": "type", "instructions": "",
            "choices": [{"value": v, "instructions": ""} for v in MESSAGE_TYPE_LABELS],
        }
        body["policy"] = {"kind": "choice-confidence", "minConfidence": 0.0}
    body["fit"] = {"outcome": "within-margin", "qualityTarget": "T=0.94"}
    assert _load(tmp_path, body).role == role
    body["fit"] = {"outcome": "within-margin", "qualityTarget": "D_c<=0"}
    _refused(tmp_path, body, match="qualityTarget")


def test_candidate_load_refuses_a_file_whose_id_disagrees_with_its_name(tmp_path):
    _refused(tmp_path, _with(("candidateId",), "guard-laya-band-v2"), match="candidateId")


@pytest.mark.parametrize("candidate_id", ["../escape", "a/b", "", ".", "..", "guard\n"])
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
        (("fitManifest", "sha256"), "0" * 64 + "\n"),
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


@pytest.mark.parametrize("edge", [0.0, 0.5, 1.0])
def test_candidate_load_accepts_an_empty_band_with_a_hand_off_model(tmp_path, edge):
    """Plan3 §3.2, the one-way rule: an empty band with a hand-off model is an overflow-only
    composite — its only hand-offs are over-limit items — and the runtime's own shape."""
    assert _load(tmp_path, _band(edge, edge)).handoffModel == "qwen/qwen3-4b-2507"


def test_candidate_load_refuses_a_band_with_no_hand_off_model(tmp_path):
    """Plan3 §3.2: a policy that can hand off needs a hand-off model — `handoffModel: null`
    requires `l == h`."""
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


def test_candidate_load_accepts_c_zero_with_a_hand_off_model(tmp_path):
    """Plan3 §3.2: `minConfidence == 0` never hands off on policy, so it is the choice-side
    overflow-only composite."""
    assert _load(tmp_path, _choice_candidate(0.0)).handoffModel == "qwen/qwen3-4b-2507"


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


# --- question_spec_hash: hand-derived vectors (impl review Pass 6, P6-m1) -----------------------

#: `(question as a candidate file would hold it, its canonical JSON, the expected sha256)`.
#: Each question's keys are deliberately out of order — top level and inside a `choices` object —
#: and its text carries non-ASCII (an em dash, Portuguese accents). The canonical string is
#: written by hand; the hash was derived from it with `printf '%s' '<canonical>' | sha256sum`,
#: never by running `question_spec_hash`.
_QUESTION_SPEC_VECTORS = [
    (
        {
            "type": "choice",
            "name": "tipo",
            "instructions": "Qual é o tipo da mensagem — pedido ou dúvida?",
            "choices": [
                {"instructions": "o cliente quer comprar", "value": "pedido"},
                {"value": "dúvida", "instructions": "pergunta sobre um produto"},
            ],
        },
        '{"choices":[{"instructions":"o cliente quer comprar","value":"pedido"},'
        '{"instructions":"pergunta sobre um produto","value":"dúvida"}],'
        '"instructions":"Qual é o tipo da mensagem — pedido ou dúvida?","name":"tipo",'
        '"type":"choice"}',
        "13bbec5c87a3ffd2e371b88f06ec9bc3a0453401a53e16e8c02e4fbcddae4875",
    ),
    (
        {
            "type": "predicate",
            "name": "ação",
            "instructions": "O usuário nomeou o serviço — sim ou não?",
        },
        '{"instructions":"O usuário nomeou o serviço — sim ou não?","name":"ação",'
        '"type":"predicate"}',
        "73cc51e7ad0bb8679086e2034ba10b397e6f2bd3c2b53ad02885932c21f6bdb0",
    ),
]


@pytest.mark.parametrize(
    ("question", "canonical", "expected"), _QUESTION_SPEC_VECTORS, ids=["choice", "predicate"]
)
def test_question_spec_hash_matches_the_hand_derived_vectors(question, canonical, expected):
    """These vectors are the formula's authoritative home until a plan states it (P6-m1).

    **The formula:** `questionSpecHash = sha256(utf8(canonical_json(question))).hexdigest()`,
    where `canonical_json` is JSON with **keys sorted at every level** (a `choices` object's keys
    included; list order kept), **no whitespace** (`","` and `":"` separators) and **non-ASCII
    kept raw** — encoded as UTF-8 bytes, never `\\uXXXX`-escaped. It is `input_hash`'s canonical
    form, applied to the candidate's whole question spec."""
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == expected  # the vector itself
    assert question_spec_hash(question) == expected
    reordered = dict(reversed(list(question.items())))
    assert question_spec_hash(reordered) == expected


# --- load_fit_manifest and leaked_items (v1.3 §3.4, §3.5, AC-3; U2.1) ---------------------------
#
# The manifest file is "the sorted list of `input_hash` values" (v1.3 §3.4): one lowercase-hex
# sha256 per line, each `\n`-terminated, strictly ascending; `size` is the entry count.

_H1, _H2, _H3 = sorted(hashlib.sha256(f"in-{n}".encode()).hexdigest() for n in (1, 2, 3))


def _manifest(root: Path, content: bytes, **declared: Any):
    (root / "candidates").mkdir(exist_ok=True)
    (root / "candidates" / "m.fit.txt").write_bytes(content)
    body = {"file": "m.fit.txt", "sha256": hashlib.sha256(content).hexdigest(), "size": None}
    body.update(declared)
    return decisions.load_fit_manifest(root, body)


def test_a_well_formed_manifest_loads_with_its_measured_hash_and_entry_count(tmp_path):
    content = f"{_H1}\n{_H2}\n{_H3}\n".encode()
    manifest = _manifest(tmp_path, content, size=3)
    assert manifest == decisions.FitManifest(
        hashes=frozenset({_H1, _H2, _H3}),
        sha256=hashlib.sha256(content).hexdigest(),
        size=3,
    )


def test_an_empty_manifest_is_zero_inputs_not_a_refusal(tmp_path):
    manifest = _manifest(tmp_path, b"", size=0)
    assert (manifest.hashes, manifest.size) == (frozenset(), 0)


def test_a_manifest_whose_bytes_do_not_match_the_declared_sha256_refuses(tmp_path):
    with pytest.raises(CandidateError, match="declared fitManifest.sha256"):
        _manifest(tmp_path, f"{_H1}\n".encode(), size=1, sha256="0" * 64)


def test_a_manifest_whose_entry_count_is_not_the_declared_size_refuses(tmp_path):
    """The count, never the byte length: 2 entries are 130 bytes."""
    for size in (1, 3, 130):
        with pytest.raises(CandidateError, match="declared fitManifest.size"):
            _manifest(tmp_path, f"{_H1}\n{_H2}\n".encode(), size=size)


@pytest.mark.parametrize(
    "content,reason",
    [
        (f"{_H2}\n{_H1}\n", "strictly ascending"),
        (f"{_H1}\n{_H1}\n", "strictly ascending"),
        (f"{_H1}\n{_H2}", "terminating newline"),
        (f"{_H1}\n\n{_H2}\n", "line 2"),
        (f"{_H1.upper()}\n", "line 1"),
        (f"{_H1} \n", "line 1"),
        ("not-a-hash\n", "line 1"),
    ],
    ids=["unsorted", "repeated", "no-final-newline", "blank-line", "upper-case", "trailing-space",
         "not-a-hash"],
)
def test_a_malformed_manifest_refuses(tmp_path, content, reason):
    with pytest.raises(CandidateError, match=reason):
        _manifest(tmp_path, content.encode(), size=content.count("\n"))


_FIT_MANIFEST_VECTORS = _rows("fit-manifest.jsonl")
_FIT_MANIFEST_ROWS = [r for r in _FIT_MANIFEST_VECTORS if "case" in r]


def test_the_fit_manifest_vectors_state_their_format_and_cover_every_refusal_rule():
    """P8-m1: the file is the cross-component contract U6's `fit.py` byte-copies, so its header
    row states the format, and every refusal rule has a row (shrink-guard on the vector file)."""
    assert "_format" in _FIT_MANIFEST_VECTORS[0] and "case" not in _FIT_MANIFEST_VECTORS[0]
    assert len(_FIT_MANIFEST_ROWS) == len(_FIT_MANIFEST_VECTORS) - 1
    assert {r["rule"] for r in _FIT_MANIFEST_ROWS if not r["accept"]} == {
        "unsorted", "repeated", "uppercase", "no-final-newline", "crlf", "wrong-size",
        "wrong-sha256",
    }
    assert any(r["accept"] and r["text"] == "" for r in _FIT_MANIFEST_ROWS)


@pytest.mark.parametrize("row", _FIT_MANIFEST_ROWS, ids=[r["case"] for r in _FIT_MANIFEST_ROWS])
def test_load_fit_manifest_against_the_vectors(tmp_path, row):
    """`fit-manifest.jsonl` (P8-m1): the row's `text` is written as the file's exact UTF-8 bytes,
    declared with the row's hand-computed `sha256` and `size`."""
    (tmp_path / "candidates").mkdir()
    (tmp_path / "candidates" / "m.fit.txt").write_bytes(row["text"].encode("utf-8"))
    declared = {"file": "m.fit.txt", "sha256": row["sha256"], "size": row["size"]}
    if row["accept"]:
        assert decisions.load_fit_manifest(tmp_path, declared) == decisions.FitManifest(
            hashes=frozenset(row["entries"]), sha256=row["sha256"], size=row["size"]
        )
        assert len(row["entries"]) == row["size"]
    else:
        with pytest.raises(CandidateError) as caught:
            decisions.load_fit_manifest(tmp_path, declared)
        assert row["refusal"] in str(caught.value)


@pytest.mark.parametrize("name", ["../m.fit.txt", "sub/m.fit.txt", "..", "", None])
def test_a_manifest_file_outside_candidates_is_refused_before_any_read(tmp_path, name):
    with pytest.raises(CandidateError, match="plain file name"):
        decisions.load_fit_manifest(tmp_path, {"file": name, "sha256": "0" * 64, "size": 0})


def test_a_missing_manifest_file_refuses(tmp_path):
    with pytest.raises(CandidateError, match="cannot read fit manifest"):
        decisions.load_fit_manifest(tmp_path, {"file": "m.fit.txt", "sha256": "0" * 64, "size": 0})


def _canonical_input_hash(item: dict[str, Any]) -> str:
    """v1.3 §3.5's `input_hash` for `guard-judge`, transcribed here rather than imported."""
    fields = {k: item[k] for k in ("condition", "understanding", "turns", "path") if k in item}
    text = json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_leaked_items_names_every_overlapping_item_of_either_stratum_in_pack_order():
    items = [
        {"itemId": "e1", "stratum": "eval", "condition": "c", "turns": [], "path": "turns"},
        {"itemId": "s1", "stratum": "side-check", "condition": "ç—x", "turns": [], "path": "turns"},
        {"itemId": "e2", "stratum": "eval", "condition": "d", "turns": [], "path": "turns"},
    ]
    overlap = sorted(_canonical_input_hash(items[i]) for i in (1, 2))
    manifest = decisions.FitManifest(hashes=frozenset(overlap), sha256="0" * 64, size=2)
    assert decisions.leaked_items(items, "guard-judge", manifest) == ["s1", "e2"]
    clean = decisions.FitManifest(hashes=frozenset({_H1}), sha256="0" * 64, size=1)
    assert decisions.leaked_items(items, "guard-judge", clean) == []


def test_leakage_is_keyed_on_input_fields_only_never_the_label_or_the_item_id():
    """The same input under another id and label is the same input (v1.3 §3.5)."""
    item = {"itemId": "a", "expected": True, "condition": "c", "turns": [], "path": "turns"}
    twin = {**item, "itemId": "b", "expected": False, "stratum": "eval"}
    manifest = decisions.FitManifest(
        hashes=frozenset({_canonical_input_hash(item)}), sha256="0" * 64, size=1
    )
    assert decisions.leaked_items([twin], "guard-judge", manifest) == ["b"]
