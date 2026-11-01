"""The decision points' definitions under `points/` (typed-decision-models U5.3; plan
`docs/plans/typed-decision-models4.md` §4.1, §3.2; method note
`docs/plans/typed-decision-models-ml.md` §Q1.1–§Q1.5).

Expected values are written here from their sources, not read from `points/`:
- answer spaces and answer keys: v1.3 `docs/plans/typed-decision-models.md` §3.2's table and
  requirements FR-10 (the five message types);
- input fields: v1.3 §3.5 (`guard-judge`, `join-in-judge`) and plan2 §3.1
  (`message-type-classifier`), and `decisionmodels.hashing` wherever it already defines a role;
- cells and sizes: note §Q1.3's table.
"""

from __future__ import annotations

import dataclasses
import json
import re
import unicodedata
from pathlib import Path

import pytest
from test_fr8_isolation import FORBIDDEN_FRAGMENTS

from decisionmodels import hashing, render, seeds
from decisionmodels.points import (
    POINTS_DIR,
    ROLES,
    input_field_errors,
    load_point,
    screen_tokens,
    seed_binding_errors,
)

ANSWERS = {
    "guard-judge": ("predicate", "advance", (True, False)),
    "join-in-judge": ("predicate", "reply", (True, False)),
    "message-type-classifier": (
        "choice",
        "type",
        ("order", "question", "complaint", "escalation", "other"),
    ),
}
PLANNED_INPUT_FIELDS = {
    "guard-judge": ("condition", "understanding", "turns", "path"),
    "join-in-judge": ("agentName", "priorTurns", "newMessage"),
    "message-type-classifier": ("text", "priorTurn"),
}
CELLS = {  # (cell count, eval per cell, eval floor per cell)
    "guard-judge": (4, 100, 50),
    "join-in-judge": (2, 100, 100),
    "message-type-classifier": (5, 60, 52),
}

GUARD_UNDERSTANDING_ITEM = {
    "condition": "the user has stated a budget for the request",
    "path": "understanding",
    "understanding": {
        "request": "Find a sofa bed for a small flat",
        "known": ["budget: under 900", "colour: grey or green"],
        "missing": [],
    },
    "turns": [],
}
GUARD_TURNS_ITEM = {
    "condition": "the user has chosen one of the options that were offered",
    "path": "turns",
    "understanding": {},
    "turns": [
        {"role": "assistant", "displayName": "Pip", "text": "Red or white roses?"},
        {"role": "user", "displayName": "Marta", "text": "White, please."},
    ],
}


# --- answer space, fields, cells -----------------------------------------------------------------


@pytest.mark.parametrize("role", ROLES)
def test_the_answer_space_matches_the_contract(role: str) -> None:
    point = load_point(role)
    kind, key, expected = ANSWERS[role]
    assert (point.kind, point.definition["answerKey"], point.expected) == (kind, key, expected)


@pytest.mark.parametrize("role", ROLES)
def test_the_input_fields_match_the_leakage_key(role: str) -> None:
    wanted = hashing.INPUT_FIELDS_BY_ROLE.get(role, PLANNED_INPUT_FIELDS[role])
    assert load_point(role).input_fields == wanted == PLANNED_INPUT_FIELDS[role]


def test_guard_inputs_equal_hashings_row_as_a_set() -> None:
    assert set(hashing.INPUT_FIELDS_BY_ROLE["guard-judge"]) == set(
        load_point("guard-judge").input_fields
    )


@pytest.mark.xfail(
    strict=True,
    raises=KeyError,
    reason="hashing.py has no row for this role until U5.2b copies U4's (plan5 §3.4, P3-F6); "
    "the XPASS then forces this xfail and PLANNED_INPUT_FIELDS' fallback to be deleted",
)
@pytest.mark.parametrize("role", ("join-in-judge", "message-type-classifier"))
def test_hashing_defines_the_new_roles_with_the_points_fields(role: str) -> None:
    """Set comparison, so tuple order between `point.json` and model-bench's row is not churn. A
    row whose fields differ is an AssertionError, which this xfail does not absorb: red."""
    assert set(hashing.INPUT_FIELDS_BY_ROLE[role]) == set(load_point(role).input_fields)


@pytest.mark.parametrize("role", ROLES)
def test_cells_and_sizes_match_the_note(role: str) -> None:
    cells = load_point(role).definition["cells"]
    assert (cells["count"], cells["evalPerCell"], cells["evalFloorPerCell"]) == CELLS[role]


def test_only_guard_has_a_design_stratum_the_evidence_kind() -> None:
    assert load_point("guard-judge").design["designStratumAxis"] == "evidenceKind"
    assert load_point("join-in-judge").design["designStratumAxis"] is None
    assert load_point("message-type-classifier").design["designStratumAxis"] is None


def test_the_evidence_kinds_are_the_guard_paths() -> None:
    point = load_point("guard-judge")
    axis = next(a for a in point.design["stratified"] if a["axis"] == "evidenceKind")
    kinds = [entry["value"] for entry in axis["values"]]
    assert kinds == point.definition["inputFields"]["path"]["values"] == ["understanding", "turns"]


@pytest.mark.parametrize("role", ROLES)
def test_the_intended_answer_ranges_over_the_whole_answer_space(role: str) -> None:
    point = load_point(role)
    axis = next(a for a in point.design["stratified"] if a["axis"] == "intendedAnswer")
    assert tuple(entry["value"] for entry in axis["values"]) == point.expected


def test_an_unknown_role_is_refused() -> None:
    with pytest.raises(ValueError, match="no decision point"):
        load_point("entity-matcher")


# --- guard: schema against render.py and hashing.py ----------------------------------------------


@pytest.mark.parametrize("item", [GUARD_UNDERSTANDING_ITEM, GUARD_TURNS_ITEM])
def test_a_schema_valid_guard_item_renders_and_hashes(item: dict) -> None:
    assert input_field_errors(load_point("guard-judge"), item) == []
    text = render.render_guard_item(item)
    assert text.startswith(f"CONDITION: {item['condition']}")
    assert hashing.input_hash(item, "guard-judge") != hashing.input_hash(
        {**item, "condition": item["condition"] + "."}, "guard-judge"
    )


def test_the_understanding_item_renders_its_state_block() -> None:
    assert "CURRENT STATE:" in render.render_guard_item(GUARD_UNDERSTANDING_ITEM)
    assert "RECENT TURNS" not in render.render_guard_item(GUARD_UNDERSTANDING_ITEM)


def test_the_turns_item_renders_display_names_as_speakers() -> None:
    text = render.render_guard_item(GUARD_TURNS_ITEM)
    assert "CURRENT STATE" not in text
    turns = "RECENT TURNS (context only):\nPip: Red or white roses?\nMarta: White, please."
    assert text.endswith(turns)


def test_guard_turns_fit_the_renderers_window() -> None:
    """Note §Q1.1: turns seeds run 1–10, exercising the renderer's last-6 cut."""
    schema = load_point("guard-judge").definition["inputFields"]["turns"]
    assert schema["maxItems"] == 10 > render.RECENT_TURNS_KEPT


# --- the schema checker --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        (lambda i: i.update(itemId="x"), "unexpected field 'itemId'"),
        (lambda i: i.pop("path"), "missing field 'path'"),
        (lambda i: i.update(path="summary"), "is not one of"),
        (lambda i: i.update(condition=""), "shorter than"),
        (lambda i: i.update(turns="none"), "expected a list"),
        (lambda i: i.update(understanding=None), "null is not allowed"),
        (lambda i: i["understanding"].update(intent="x"), "unexpected field 'intent'"),
        (lambda i: i["understanding"].pop("missing"), "missing field 'missing'"),
        (lambda i: i["understanding"].update(known=[1]), "expected a string"),
    ],
)
def test_a_departing_guard_item_is_reported(change, fragment: str) -> None:
    item = json.loads(json.dumps(GUARD_UNDERSTANDING_ITEM))
    change(item)
    errors = input_field_errors(load_point("guard-judge"), item)
    assert any(fragment in e for e in errors), errors


def test_too_many_turns_and_a_bad_turn_role_are_reported() -> None:
    turn = {"role": "system", "displayName": "X", "text": "t"}
    item = {**GUARD_TURNS_ITEM, "turns": [turn] * 11}
    errors = input_field_errors(load_point("guard-judge"), item)
    assert any("more than 10" in e for e in errors)
    assert any("'system' is not one of" in e for e in errors)


def test_message_type_items_take_a_null_or_full_prior_turn() -> None:
    point = load_point("message-type-classifier")
    assert input_field_errors(point, {"text": "Hi!", "priorTurn": None}) == []
    full = {"text": "Yes, the blue one", "priorTurn": {"speaker": "Pip", "text": "Blue or grey?"}}
    assert input_field_errors(point, full) == []
    assert input_field_errors(point, {"text": "Hi!", "priorTurn": {}})
    assert input_field_errors(point, {"text": "Hi!"})


def test_join_in_new_message_is_always_from_a_person() -> None:
    point = load_point("join-in-judge")
    item = {
        "agentName": "Nova",
        "priorTurns": [],
        "newMessage": {"role": "user", "displayName": "Ana", "text": "Morning all"},
    }
    assert input_field_errors(point, item) == []
    item["newMessage"]["role"] = "assistant"
    assert input_field_errors(point, item)


def test_a_non_object_item_is_reported() -> None:
    assert input_field_errors(load_point("guard-judge"), ["condition"]) == [
        "item: expected an object"
    ]


# --- seed bindings -------------------------------------------------------------------------------


def _guard_axes(
    kind: str, condition: str, turn_count: int | None, form: str | None = None
) -> dict:
    if form is None and kind == "understanding":
        form = "empty"
    return {"evidenceKind": kind, "condition": condition, "turnCount": turn_count,
            "missingForm": form}


def test_a_guard_item_bound_to_its_seed_passes() -> None:
    point = load_point("guard-judge")
    understanding = _guard_axes("understanding", GUARD_UNDERSTANDING_ITEM["condition"], None)
    turns = _guard_axes("turns", GUARD_TURNS_ITEM["condition"], 2)
    assert seed_binding_errors(point, GUARD_UNDERSTANDING_ITEM, understanding) == []
    assert seed_binding_errors(point, GUARD_TURNS_ITEM, turns) == []


@pytest.mark.parametrize(
    ("axes", "fragment"),
    [
        (_guard_axes("turns", GUARD_TURNS_ITEM["condition"], 3), "turnCount"),
        (_guard_axes("turns", "another condition", 2), "condition"),
        (_guard_axes("understanding", GUARD_TURNS_ITEM["condition"], None), "evidenceKind"),
    ],
)
def test_a_guard_item_off_its_seed_is_reported(axes: dict, fragment: str) -> None:
    errors = seed_binding_errors(load_point("guard-judge"), GUARD_TURNS_ITEM, axes)
    assert any(error.axis == fragment for error in errors), errors
    assert all(fragment in error.message for error in errors if error.axis == fragment)


#: Note v1.11 §Q1.1: each form fixes whether `missing` is empty.
MISSING_EMPTY = {"empty": True, "indirect": True, "unlisted": True, "ambiguous": True,
                 "immaterial": False, "listed": False}


@pytest.mark.parametrize("form", sorted(MISSING_EMPTY))
@pytest.mark.parametrize("missing", [[], ["gift wrapping"]])
def test_a_guard_item_obeys_its_missing_form(form: str, missing: list) -> None:
    """Per-item obedience (note v1.11 §Q1.1): one obeying and one disobeying case per form."""
    item = json.loads(json.dumps(GUARD_UNDERSTANDING_ITEM))
    item["understanding"]["missing"] = missing
    axes = _guard_axes("understanding", item["condition"], None, form)
    errors = seed_binding_errors(load_point("guard-judge"), item, axes)
    obeys = (not missing) == MISSING_EMPTY[form]
    assert (errors == []) == obeys, errors
    if not obeys:
        assert len(errors) == 1, errors
        error = errors[0]
        assert error.axis == "missingForm" and error.binding["emptinessAxis"] == "missingForm"
        assert form in error.message


def test_a_turns_item_is_not_bound_by_a_missing_form() -> None:
    axes = _guard_axes("turns", GUARD_TURNS_ITEM["condition"], 2)
    assert axes["missingForm"] is None
    assert seed_binding_errors(load_point("guard-judge"), GUARD_TURNS_ITEM, axes) == []


@pytest.mark.parametrize("form", sorted(MISSING_EMPTY))
def test_a_wrong_kind_item_is_an_evidence_kind_error_only(form: str) -> None:
    """Pass 4 P4-F1: a `turns` item against an `understanding` seed fails the `evidenceKind`
    bindings (path, `understanding` empty, `turns` present) and never also its `missingForm`, so
    the ingester cannot count it as a form error."""
    axes = _guard_axes("understanding", GUARD_TURNS_ITEM["condition"], None, form)
    errors = seed_binding_errors(load_point("guard-judge"), GUARD_TURNS_ITEM, axes)
    assert [(e.axis, e.binding["field"]) for e in errors] == [
        ("evidenceKind", "path"), ("evidenceKind", "understanding"), ("evidenceKind", "turns")
    ]


def test_an_undefined_missing_form_is_a_data_error() -> None:
    axes = _guard_axes("understanding", GUARD_TURNS_ITEM["condition"], None, "sideways")
    with pytest.raises(ValueError, match="no emptiness rule"):
        seed_binding_errors(load_point("guard-judge"), GUARD_TURNS_ITEM, axes)


def test_the_obedience_rule_covers_every_drawn_form() -> None:
    bindings = load_point("guard-judge").definition["seedBindings"]
    binding = next(b for b in bindings if "emptinessAxis" in b)
    assert set(binding["emptyFor"]) == {f for f, empty in MISSING_EMPTY.items() if empty}
    assert set(binding["nonEmptyFor"]) == {f for f, empty in MISSING_EMPTY.items() if not empty}
    # The emptiness binding is silent on an empty field because this binding reports it.
    assert any(b.get("presentWhen") and b["field"] == binding["field"] for b in bindings)


def test_a_message_type_prior_turn_follows_the_thread_position() -> None:
    point = load_point("message-type-classifier")
    opening, follow_up = {"threadPosition": "opening"}, {"threadPosition": "follow-up"}
    bare = {"text": "Hi", "priorTurn": None}
    with_prior = {"text": "Yes", "priorTurn": {"speaker": "Pip", "text": "Blue?"}}
    assert seed_binding_errors(point, bare, opening) == []
    assert seed_binding_errors(point, with_prior, follow_up) == []
    for item, axes in ((with_prior, opening), (bare, follow_up)):
        errors = seed_binding_errors(point, item, axes)
        assert [(e.axis, e.binding["field"]) for e in errors] == [("threadPosition", "priorTurn")]


def test_a_binding_error_names_the_binding_that_failed() -> None:
    """Plan5 §3.5: the ingester classifies on the binding, not the message. An item off its seed
    on two bindings reports each with its own binding object from `point.json`."""
    point = load_point("guard-judge")
    item = json.loads(json.dumps(GUARD_UNDERSTANDING_ITEM))
    item["understanding"]["missing"] = ["gift wrapping"]
    axes = _guard_axes("understanding", "another condition", None, "empty")
    errors = seed_binding_errors(point, item, axes)
    bindings = point.definition["seedBindings"]
    assert [bindings.index(e.binding) for e in errors] == [0, 5]
    assert [e.axis for e in errors] == ["condition", "missingForm"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        errors[0].message = "x"  # type: ignore[misc]


# --- screen token lists --------------------------------------------------------------------------


@pytest.mark.parametrize("role", ROLES)
def test_the_screen_lists_cover_the_notes_hits(role: str) -> None:
    """Note §Q1.2 "Hits": `near-miss` anywhere; `intended`, `difficulty` and the answer names in
    a meta position."""
    point = load_point(role)
    tokens = screen_tokens(point)
    assert "near-miss" in tokens["anywhere"]
    names = {json.dumps(v).strip('"') for v in point.expected}
    assert {"intended", "difficulty", *names} <= set(tokens["metaPosition"])


def test_the_guard_screen_lists_the_six_missing_form_names() -> None:
    """Note v1.10 §Q1.1 "Guard `missing` forms": the six form names join `metaPosition`."""
    forms = {"empty", "immaterial", "indirect", "listed", "unlisted", "ambiguous"}
    assert forms <= set(screen_tokens(load_point("guard-judge"))["metaPosition"])


# --- templates and rubrics carry no seed ---------------------------------------------------------

_PLACEHOLDER = re.compile(r"\{[A-Za-z_]+\}")
_RUBRIC_FORBIDDEN_WORDS = re.compile(
    r"\b(inten[dt]\w*|difficult\w*|near[- ]miss|seeds?|strat(um|a)|axis|axes"
    r"|synthetic|generat\w*)\b",
    re.IGNORECASE,
)


def _axis_texts_and_ids(role: str) -> list[str]:
    design = load_point(role).design
    entries = [entry for axis in design["balanced"] for entry in axis["values"]]
    entries += [entry for axis in design["conditional"] for case in axis["cases"]
                for entry in case["values"]]
    return sorted({value for entry in entries if "id" in entry
                   for value in (entry["id"], entry["text"])})


def _axis_keys(role: str) -> set[str]:
    return set(seeds.draw_seeds(role, 3, 1, design_stratum=None).seeds[0].axes)


@pytest.mark.parametrize("role", ROLES)
def test_no_template_carries_a_seed_value(role: str) -> None:
    point = load_point(role)
    for text in (point.rubric, point.generator_instructions):
        assert not seeds.ID_PATTERN.search(text)
        assert not _PLACEHOLDER.search(text)
        lowered = text.lower()
        assert [v for v in _axis_texts_and_ids(role) if v.lower() in lowered] == []


@pytest.mark.parametrize("role", ROLES)
def test_no_rubric_mentions_seed_axes_intent_or_difficulty(role: str) -> None:
    rubric = load_point(role).rubric
    assert _RUBRIC_FORBIDDEN_WORDS.findall(rubric) == []
    # An axis that is also an input field (guard's `condition`) is rendered text the labeller sees.
    hidden = _axis_keys(role) - set(load_point(role).input_fields)
    assert [key for key in hidden if re.search(rf"\b{key}\b", rubric)] == []


@pytest.mark.parametrize("role", ROLES)
def test_a_rubric_defines_every_answer_with_boundary_cases(role: str) -> None:
    point = load_point(role)
    for value in point.expected:
        assert re.search(rf"^### {json.dumps(value).strip(chr(34))}\b", point.rubric_body, re.M)
    assert "\n## Boundary cases\n" in point.rubric_body
    assert '"answer"' in point.rubric and "bodySha256" in point.rubric


@pytest.mark.parametrize("role", ROLES)
def test_the_generator_template_explains_every_seed_field_and_item_field(role: str) -> None:
    point = load_point(role)
    text = point.generator_instructions
    text_axes = {key for key in _axis_keys(role) if not key.endswith("Id")}
    for name in sorted(text_axes | {"key"} | set(point.input_fields)):
        assert f"`{name}`" in text, name


@pytest.mark.parametrize("role", ROLES)
def test_the_generator_template_names_no_id_field(role: str) -> None:
    """Note v1.10 §Q1.2 "Blind": no `*Id` field reaches the generator, so the template does not
    describe one."""
    id_keys = {key for key in _axis_keys(role) if key.endswith("Id")} | {"seedId"}
    text = load_point(role).generator_instructions
    assert [key for key in sorted(id_keys) if key in text] == []
    assert "no commentary" in text
    assert '"item"' in text and "bodySha256" in text


# --- note v1.10 rulings in the rubric and template text -----------------------------------------

#: role -> phrases each ruling must keep (note v1.10 §Q1.1 "Message-type rubric cases", "Join-in
#: cue given the answer", "The assistant's display name").
RUBRIC_RULINGS = {
    "message-type-classifier": (
        "unrelated to the business",
        "within the business's scope",
        "Asking to cancel, change or return because",
        "Threatening to cancel",
        "Cancelling with nothing wrong",
        "changed their mind",
    ),
    "join-in-judge": (
        "addressed to the agent by name that invites an answer",
        'A greeting to the room ("Morning everyone!")',
    ),
    "guard-judge": ("`Assistant`", "`<business> Assistant`"),
}


@pytest.mark.parametrize("role", ROLES)
def test_the_rubric_carries_the_notes_rulings(role: str) -> None:
    body = load_point(role).rubric_body
    for phrase in RUBRIC_RULINGS[role]:
        assert phrase in body, phrase


@pytest.mark.parametrize("role", ("guard-judge", "message-type-classifier"))
def test_the_generator_names_the_assistant_as_production_does(role: str) -> None:
    text = load_point(role).generator_body
    assert "`Assistant`" in text and "`<business> Assistant`" in text
    assert "never a person's first name" in text


def test_the_guard_generator_keeps_the_condition_and_the_business() -> None:
    """Note v1.10 §Q1.1 "Guard conditions": adapt the situation, never the sentence or business."""
    text = load_point("guard-judge").generator_body
    assert "Never change the condition sentence or the business." in text


@pytest.mark.parametrize("form", ["empty", "immaterial", "indirect", "listed", "unlisted",
                                  "ambiguous"])
def test_the_guard_generator_states_each_missing_form_rule(form: str) -> None:
    assert f"  - `{form}`: " in load_point("guard-judge").generator_body


@pytest.mark.parametrize("form", sorted(MISSING_EMPTY))
def test_a_form_rule_says_missing_is_empty_exactly_when_its_binding_does(form: str) -> None:
    """ML Pass 2 P2-m1: the generator's rule must not let `missing` hold anything the binding
    rejects, so each form bound to an empty list says so in its own rule."""
    body = load_point("guard-judge").generator_body
    start = body.index(f"  - `{form}`: ")
    end = min(i for i in (body.find("\n  - ", start + 1), body.find("\n\n", start)) if i != -1)
    rule = " ".join(body[start:end].split())
    assert ("`missing` is `[]`" in rule) == MISSING_EMPTY[form], rule


# --- FR-8 and note §Q1.4 over the data files -----------------------------------------------------


def _points_texts() -> list[tuple[str, str]]:
    return [(str(p), p.read_text(encoding="utf-8")) for p in sorted(POINTS_DIR.rglob("*"))
            if p.is_file()]


def test_the_points_data_is_all_scanned() -> None:
    names = {Path(p).relative_to(POINTS_DIR).as_posix() for p, _ in _points_texts()}
    assert {f"{r}/{f}" for r in ROLES for f in ("point.json", "rubric.md", "generator.md")} <= names


def test_no_points_file_names_a_real_data_reach() -> None:
    """`points/` is data, so the AST allowlist does not apply, but its text is sent to subagents:
    it must not point them at falkor-chat, FalkorDB/Redis or the side-check pack."""
    found = [
        (path, name)
        for path, text in _points_texts()
        for name, pattern in FORBIDDEN_FRAGMENTS.items()
        if pattern.search(text.lower())
    ]
    assert found == []


_MODEL_BENCH = Path(__file__).resolve().parents[2] / "model-bench"
_SIDE_CHECK = _MODEL_BENCH / "packs" / "guard-judge-understanding" / "items.jsonl"


def _tokens(text: str) -> list[str]:
    """Note §Q1.3: NFKC, lowercase, every non-letter-or-digit to a space, split."""
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(c if c.isalpha() or c.isdigit() else " " for c in text).split()


def _shingles(text: str, n: int = 8) -> set[tuple[str, ...]]:
    tokens = _tokens(text)
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _side_check_items(model_bench: Path) -> Path:
    """The side-check items file. Skips only when `model-bench/` itself is absent (v1.3 §3.2.2's
    rule); a present `model-bench/` without the file fails, so a moved pack cannot turn the
    note §Q1.4 gate into a skip (Pass 3 P3-F5)."""
    if not model_bench.is_dir():
        pytest.skip("model-bench/ is absent from this checkout (v1.3 §3.2.2 skip rule)")
    items = model_bench / _SIDE_CHECK.relative_to(_MODEL_BENCH)
    assert items.is_file(), f"model-bench/ is present but {items} is not"
    return items


def test_a_moved_side_check_pack_fails_rather_than_skips(tmp_path: Path) -> None:
    """`pytest.skip` raises a `BaseException`, which `pytest.raises(AssertionError)` would let
    through as a skip of this very test; so the skip is caught and turned into a failure."""
    try:
        _side_check_items(tmp_path)
    except pytest.skip.Exception:
        pytest.fail("a present model-bench/ without the side-check items skipped")
    except AssertionError as error:
        assert "is present but" in str(error)
    else:
        pytest.fail("a present model-bench/ without the side-check items passed")


def test_no_points_text_shares_an_eight_token_span_with_a_side_check_item() -> None:
    """Note §Q1.4: the 85 human-curated items never appear in any prompt. Reading them from
    `tests/` is not an FR-8 reach (the scan covers `decisionmodels/`)."""
    side_check = _side_check_items(_MODEL_BENCH)
    side: set[tuple[str, ...]] = set()
    for line in side_check.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            evidence = json.dumps(item.get("understanding"), ensure_ascii=False)
            evidence += " " + " ".join(t.get("text", "") for t in item.get("turns") or [])
            side |= _shingles(evidence)
    assert side
    ours = set().union(*(_shingles(text) for _, text in _points_texts()))
    assert ours & side == set()
