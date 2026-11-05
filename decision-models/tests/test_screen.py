"""The generated-text screen and form obedience (typed-decision-models U5.8a).

Rows: plan4 `docs/plans/typed-decision-models4.md` §5 U5.8 "Screen"; plan5
`typed-decision-models5.md` §4 U5.8 "Obedience" and "Screen" (the pure parts). The campaign-level
rows (regeneration, the cap, the halt) are `test_generation.py`.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from decisionmodels import seeds
from decisionmodels.points import ROLES, load_point, seed_binding_errors
from decisionmodels.screen import (
    BINDING_MISMATCH,
    FORM_MISMATCH,
    SCREEN,
    check_item,
    evidence_strings,
    screen_hit,
)

GUARD = load_point("guard-judge")
MESSAGE_TYPE = load_point("message-type-classifier")
JOIN_IN = load_point("join-in-judge")
FORMS = ("empty", "immaterial", "indirect", "listed", "unlisted", "ambiguous")
EMPTY_FORMS = {"empty", "indirect", "unlisted", "ambiguous"}


def turns_item(*texts: str, condition: str = "the user has stated a budget") -> dict[str, Any]:
    return {"condition": condition, "path": "turns", "understanding": {},
            "turns": [{"role": "user", "displayName": "Ana", "text": t} for t in texts]}


def understanding_item(missing: list[str], *, known: list[str] | None = None,
                       condition: str = "the user has stated a budget") -> dict[str, Any]:
    return {"condition": condition, "path": "understanding", "turns": [],
            "understanding": {"request": "Find a sofa bed", "known": known or ["colour: grey"],
                              "missing": missing}}


def obedient_item(seed: seeds.Seed) -> dict[str, Any]:
    """A guard item that obeys `seed`: its condition and kind, its turn count, its form."""
    axes = seed.axes
    if axes["evidenceKind"] == "turns":
        return turns_item(*[f"Mensagem {n}" for n in range(axes["turnCount"])],
                          condition=axes["condition"])
    missing = [] if axes["missingForm"] in EMPTY_FORMS else ["o prazo de entrega"]
    return understanding_item(missing, condition=axes["condition"])


def guard_axes(kind: str = "understanding", form: str | None = "empty",
               turn_count: int | None = None) -> dict[str, Any]:
    return {"evidenceKind": kind, "condition": "the user has stated a budget",
            "turnCount": turn_count, "missingForm": form}


# --- the hit kinds -------------------------------------------------------------------------------


@pytest.mark.parametrize(("text", "reason"), [
    ("Isso foi um near-miss claro", "token:near-miss"),
    ("NEAR-MISS", "token:near-miss"),
    ("o código dom_07 ficou na tela", "id:dom_07"),
    ("(dom_07)", "id:dom_07"),
    ("ref: cond_00.", "id:cond_00"),
    ("sd_004821937562", "id:sd_004821937562"),
])
def test_a_token_no_prose_uses_is_a_hit_anywhere(text: str, reason: str) -> None:
    assert screen_hit(GUARD, turns_item(text)) == reason


@pytest.mark.parametrize(("text", "token"), [
    ("Quero pagar (difficulty) menos", "difficulty"),
    ("Total [clear]", "clear"),
    ("Ok (intended)", "intended"),
    ("Hmm (no)", "no"),
    ("(yes)", "yes"),
    ("(true, false)", "true"),
    ("x (clear / advance) y", "clear"),
    ("x (suspend | advance)", "suspend"),
    ("Paying (difficulty: hard)", "difficulty"),
    ("Paying [intended = yes]", "intended"),
    ("[label: advance]", "advance"),
    ("Label: advance", "advance"),
    ("the answer - yes", "yes"),
    ("CATEGORY suspend", "suspend"),
    ("intent=\"false\"", "false"),
    # a stray closer earlier in the string must not cancel a later opener (P18-M1)
    ("Thanks :) (clear)", "clear"),
    ("1) first 2) (clear)", "clear"),
    ("a) (clear)", "clear"),
    ("(a)) x (clear)", "clear"),
    ("ok :) :) [difficulty]", "difficulty"),
])
def test_an_ordinary_word_in_a_meta_position_is_a_hit(text: str, token: str) -> None:
    assert screen_hit(GUARD, turns_item(text)) == f"meta:{token}"


@pytest.mark.parametrize("text", [
    "I had difficulty paying the invoice",
    "It is a plain, clear day and the answer was no problem",
    "Can you say yes to that?",
    "pedido_123 chegou",
    "opções_12 e sku_a1",
    "a clear mix (de cores)",
    "the label printer is broken",
    "open ( then clear",
    "Hello, near miss of a kind",
    # a bracket whose content is more than listed tokens is prose (note §Q1.2, 2026-11-04)
    "Where is my order (order #4411)?",
    "Not that (other than login)",
    "Use (type A) please",
    "Give me a sec (no rush)",
    "Ok (the intended answer)",
    "(sem pressa, true)",
    "a) (b clear c)",
    "(ok) difficulty",
    "(unclosed difficulty",
    "stray ) clear",
    "stray ) clear )",
    "(clear]",
    "[clear)",
    "() []",
])
def test_ordinary_prose_and_word_bounded_ids_pass(text: str) -> None:
    assert screen_hit(GUARD, turns_item(text)) is None


@pytest.mark.parametrize(("text", "token"), [
    ("(intended answer is true)", "true"),
    ("(the answer is no)", "no"),
    ("(difficulty clear)", "clear"),
    ("(clear difficulty)", "clear"),
    ("(no difficulty)", "no"),
    ("(clear.)", "clear"),
    ("(true!)", "true"),
    ("(clear?;)", "clear"),
    ("(clear/ambiguous)", "clear"),
    ("(category: x, label is advance)", "advance"),
    ("[intent suspend]", "suspend"),
    ("(intended)", "intended"),
    ("[clear | advance]", "clear"),
])
def test_the_bracket_rule_hits_note_16s_listed_strings(text: str, token: str) -> None:
    assert screen_hit(GUARD, turns_item(text)) == f"meta:{token}"


@pytest.mark.parametrize("text", [
    "(no answer)",
    "(no answer from support)",
    "(not clear)",
    "(very clear)",
    "(true story)",
    "(no rush)",
    "(listed in order)",
    "(clear and simple)",
    "(it is an order)",
    "(type A)",
    "(other than login)",
    "(order #4411)",
    "(clear/hard)",           # `hard` is not a guard token: list content, not a rule gap
    "(clear, hard)",
    "(answer later: it varies)",
    "(the difficulty of it all)",
    "(difficulty is hard)",
    "(difficulty hard)",
    "(the difficulty)",
    "(hard difficulty)",
    "(intended for you)",
    "(label printer, sad)",
    "Difficulty clear is not a bracket",
])
def test_the_bracket_rule_passes_note_16s_prose_strings(text: str) -> None:
    assert screen_hit(GUARD, turns_item(text)) is None


def test_a_keyword_reaches_two_words_forward_and_no_further_or_back() -> None:
    assert screen_hit(GUARD, turns_item("(answer is probably no)")) is None   # three words on
    assert screen_hit(GUARD, turns_item("(answer is probably, no)")) is None
    assert screen_hit(GUARD, turns_item("(no, the answer)")) is None          # backwards
    assert screen_hit(GUARD, turns_item("(answer: is no)")) == "meta:no"
    assert screen_hit(GUARD, turns_item("(difficulty a clear)")) == "meta:clear"


def test_difficulty_and_intended_are_keywords_in_brackets_only() -> None:
    assert screen_hit(GUARD, turns_item("difficulty clear")) is None
    assert screen_hit(GUARD, turns_item("intended true")) is None
    assert screen_hit(GUARD, turns_item("intent true")) == "meta:true"
    assert screen_hit(GUARD, turns_item("(intended true)")) == "meta:true"


def test_the_bracket_rule_on_the_message_type_list_passes_natural_order_text() -> None:
    for text in ("Where is my order (order #4411)?", "(other than login)", "(type A)"):
        assert screen_hit(MESSAGE_TYPE, {"text": text, "priorTurn": None}) is None
    assert screen_hit(MESSAGE_TYPE, {"text": "(order)", "priorTurn": None}) == "meta:order"
    assert screen_hit(MESSAGE_TYPE, {"text": "[type: order]", "priorTurn": None}) == "meta:type"
    assert screen_hit(MESSAGE_TYPE, {"text": "[label: order]", "priorTurn": None}) == "meta:order"


@pytest.mark.parametrize("form", FORMS)
def test_each_missing_form_name_is_a_hit_in_a_meta_position_and_a_pass_in_prose(
        form: str) -> None:
    assert screen_hit(GUARD, understanding_item([f"({form})"])) == f"meta:{form}"
    assert screen_hit(GUARD, understanding_item([f"label: {form}"])) == f"meta:{form}"
    assert screen_hit(GUARD, understanding_item([f"the dates are {form} in the text"])) is None


def test_the_guard_screen_names_the_six_forms_and_the_other_points_theirs() -> None:
    assert set(FORMS) <= set(GUARD.definition["screen"]["metaPosition"])
    assert screen_hit(MESSAGE_TYPE, {"text": "(complaint)", "priorTurn": None}) == "meta:complaint"
    assert screen_hit(MESSAGE_TYPE, {"text": "a complaint", "priorTurn": None}) is None
    join_in = {"agentName": "Pip", "priorTurns": [], "newMessage":
               {"role": "user", "displayName": "Ana", "text": "ok"}}
    assert screen_hit(JOIN_IN, join_in) is None
    join_in["newMessage"]["text"] = "cue_01"
    assert screen_hit(JOIN_IN, join_in) == "id:cue_01"


@pytest.mark.parametrize("role", ROLES)
def test_every_minted_seed_axis_id_is_a_hit_in_prose(role: str) -> None:
    ids = {value for seed in seeds.draw_seeds(role, 60, 5, design_stratum=None).seeds
           for name, value in seed.axes.items() if name.endswith("Id") and value is not None}
    assert ids
    point = load_point(role)
    for minted in sorted(ids):
        assert seeds.ID_PATTERN.fullmatch(minted), minted
        text = f"see {minted} here"
        probes = {
            "guard-judge": turns_item(text),
            "message-type-classifier": {"text": text, "priorTurn": None},
            "join-in-judge": {"agentName": "Pip", "priorTurns": [], "newMessage": {
                "role": "user", "displayName": "Ana", "text": text}},
        }
        assert screen_hit(point, probes[role]) == f"id:{minted}"


# --- where the screen looks ----------------------------------------------------------------------


def test_the_screen_reads_every_evidence_string_and_skips_enums_and_seed_fixed_text() -> None:
    item = understanding_item(["x"], known=["fine", "plain"],
                              condition="the user said (clear) near-miss")
    item["turns"] = [{"role": "user", "displayName": "Ana", "text": "olá"}]
    assert "the user said (clear) near-miss" not in evidence_strings(GUARD, item)
    assert screen_hit(GUARD, item) is None            # condition is the seed's, not the generator's
    assert evidence_strings(GUARD, item) == ["Find a sofa bed", "fine", "plain", "x", "Ana", "olá"]
    for where in (["understanding", "request"], ["understanding", "known", 1],
                  ["understanding", "missing", 0], ["turns", 0, "displayName"],
                  ["turns", 0, "text"]):
        planted = copy.deepcopy(item)
        node = planted
        for step in where[:-1]:
            node = node[step]
        node[where[-1]] = "oops near-miss"
        assert screen_hit(GUARD, planted) == "token:near-miss", where


def test_a_nested_bracket_and_an_unclosed_one() -> None:
    assert screen_hit(GUARD, turns_item("(ok, (difficulty) tambem)")) == "meta:difficulty"
    assert screen_hit(GUARD, turns_item("(ok [clear] tambem)")) == "meta:clear"
    assert screen_hit(GUARD, turns_item("(ok) difficulty")) is None
    assert screen_hit(GUARD, turns_item("(unclosed difficulty")) is None
    assert screen_hit(GUARD, turns_item("(mismatched clear]")) is None


# --- obedience -----------------------------------------------------------------------------------


@pytest.mark.parametrize("form", FORMS)
def test_an_item_that_contradicts_its_form_is_a_form_mismatch_and_an_obedient_one_passes(
        form: str) -> None:
    axes = guard_axes(form=form)
    obeys = understanding_item([] if form in EMPTY_FORMS else ["o prazo"])
    disobeys = understanding_item(["o prazo"] if form in EMPTY_FORMS else [])
    assert check_item(GUARD, axes, obeys).reasons == ()
    check = check_item(GUARD, axes, disobeys)
    assert check.reasons == (FORM_MISMATCH,) and check.primary == FORM_MISMATCH
    assert check.form == form and form in (check.detail or "")


def test_a_wrong_kind_item_with_four_errors_is_one_binding_mismatch_and_no_form_mismatch() -> None:
    item = turns_item("Oi", condition="another condition entirely")
    errors = seed_binding_errors(GUARD, item, guard_axes(form="listed"))
    assert [e.axis for e in errors] == ["condition", "evidenceKind", "evidenceKind",
                                        "evidenceKind"]
    check = check_item(GUARD, guard_axes(form="listed"), item)
    assert check.reasons == (BINDING_MISMATCH,) and check.form is None


def test_a_screen_hit_beats_a_binding_error_and_the_item_shows_every_reason() -> None:
    item = understanding_item(["x (difficulty)"], condition="a different one")
    check = check_item(GUARD, guard_axes(form="empty"), item)
    assert check.reasons == (SCREEN, BINDING_MISMATCH, FORM_MISMATCH)
    assert check.primary == SCREEN and check.detail == "meta:difficulty"
    assert check.form == "empty"


def test_a_binding_error_and_a_form_error_show_both_and_the_binding_one_wins() -> None:
    item = understanding_item(["o prazo"], condition="a different one")
    check = check_item(GUARD, guard_axes(form="empty"), item)
    assert check.reasons == (BINDING_MISMATCH, FORM_MISMATCH) and check.primary == BINDING_MISMATCH


def test_an_obedient_item_for_every_drawn_seed_passes() -> None:
    for seed in seeds.draw_seeds("guard-judge", 200, 11, design_stratum=None).seeds:
        assert check_item(GUARD, seed.axes, obedient_item(seed)).reasons == (), seed.axes


def test_a_message_type_prior_turn_error_is_a_binding_mismatch() -> None:
    check = check_item(MESSAGE_TYPE, {"threadPosition": "follow-up"},
                       {"text": "Yes", "priorTurn": None})
    assert check.reasons == (BINDING_MISMATCH,) and check.form is None
