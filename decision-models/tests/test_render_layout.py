"""Each new-ability point's rendering matches its rubric's "What an item looks like" block
(`docs/plans/typed-decision-models5.md` §3.4, U5.2b-render), and join-in's 6-turn rule.

The block is read from `points/<role>/rubric.md` itself, turned into a pattern (`<placeholder>`
becomes a capture group, a bare `...` line repeats the line above it), and a fixture item rendered
through `render.py` must match it whole, with the captured values being the fixture's own.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import pytest

from decisionmodels import exchange, render
from decisionmodels.points import load_point

_HEADING = "## What an item looks like"
_PLACEHOLDER = re.compile(r"<[^>\n]+>")

JOIN_IN_ITEM = {
    "agentName": "Pastel",
    "priorTurns": [
        {"role": "user", "displayName": "Ana", "text": "I want two cakes"},
        {"role": "assistant", "displayName": "Pastel", "text": "Chocolate or vanilla?"},
    ],
    "newMessage": {"role": "user", "displayName": "Bo", "text": "chocolate, obviously"},
}
JOIN_IN_VALUES = [
    "Pastel", "Ana", "I want two cakes", "Pastel", "Chocolate or vanilla?", "Bo",
    "chocolate, obviously",
]
MESSAGE_TYPE_ITEM = {
    "text": "Yes, the blue one",
    "priorTurn": {"speaker": "Florist Assistant", "text": "We have it in red and blue."},
}
MESSAGE_TYPE_VALUES = ["Florist Assistant", "We have it in red and blue.", "Yes, the blue one"]


def layout_lines(section: str) -> list[str]:
    """The first indented block of a rubric section, dedented: from its first indented line to the
    last indented line before the next unindented, non-blank one. Blank lines between its lines
    are kept; a later indented sample in the prose is not part of it (P26-n2)."""
    raw = section.splitlines()
    start = next((n for n, line in enumerate(raw) if line.startswith("    ")), None)
    if start is None:
        return []
    end = start
    for n in range(start, len(raw)):
        if raw[n].startswith("    "):
            end = n
        elif raw[n].strip():
            break
    return [line[4:] for line in raw[start : end + 1]]


def layout_block(role: str) -> list[str]:
    """The rubric's layout block (`layout_lines` over its "What an item looks like" section)."""
    section = load_point(role).rubric_body.split(_HEADING, 1)[1].split("\n## ", 1)[0]
    lines = layout_lines(section)
    assert lines, f"{role}'s rubric has no indented layout block"
    return lines


def layout_pattern(role: str, repeats: int) -> re.Pattern[str]:
    """The layout as a pattern, a `...` line expanded to `repeats` copies of the line above."""
    expanded: list[str] = []
    for line in layout_block(role):
        if line == "...":
            expanded.extend([expanded[-1]] * (repeats - 1))
        else:
            expanded.append(line)
    pieces = _PLACEHOLDER.split("\n".join(expanded))
    return re.compile("(.+)".join(re.escape(piece) for piece in pieces))


@pytest.mark.parametrize(
    ("role", "item", "values", "repeats", "renderer"),
    [
        ("join-in-judge", JOIN_IN_ITEM, JOIN_IN_VALUES, 2, render.render_join_in_item),
        (
            "message-type-classifier",
            MESSAGE_TYPE_ITEM,
            MESSAGE_TYPE_VALUES,
            1,
            render.render_message_type_item,
        ),
    ],
)
def test_a_rendered_item_matches_the_rubrics_layout_block(
    role: str,
    item: dict[str, Any],
    values: list[str],
    repeats: int,
    renderer: Callable[[dict[str, Any]], str],
) -> None:
    match = layout_pattern(role, repeats).fullmatch(renderer(item))
    assert match is not None
    assert list(match.groups()) == values


@pytest.mark.parametrize(
    ("role", "item"),
    [("join-in-judge", JOIN_IN_ITEM), ("message-type-classifier", MESSAGE_TYPE_ITEM)],
)
def test_the_exchange_renders_each_new_ability_point_through_its_own_renderer(
    role: str, item: dict[str, Any]
) -> None:
    assert exchange.RENDERERS[role](item) == {
        "join-in-judge": render.render_join_in_item,
        "message-type-classifier": render.render_message_type_item,
    }[role](item)
    assert exchange.RENDERERS[role] is not exchange.RENDERERS["guard-judge"]


def test_join_in_renders_only_the_last_six_of_the_eight_prior_turns_it_allows() -> None:
    schema = load_point("join-in-judge").definition["inputFields"]["priorTurns"]
    assert schema["maxItems"] == 8 > render.RECENT_TURNS_KEPT
    turns = [{"role": "user", "displayName": f"P{n}", "text": f"turn {n}"} for n in range(1, 9)]
    item = {**JOIN_IN_ITEM, "priorTurns": turns}
    rendered = render.render_join_in_item(item)
    assert [f"P{n}: turn {n}" in rendered for n in range(1, 9)] == [False] * 2 + [True] * 6


_NUMBER_WORDS = {6: "six"}


def test_the_join_in_generator_keeps_the_agents_job_inside_the_last_six_turns() -> None:
    """Plan5 §3.4: the renderer drops all but the last six prior turns while `maxItems` is 8, so
    the generator's instructions must say where the judge can see the agent's job. Pinned as the
    rule, not a phrase: the paragraph names the renderer's own number, says the judge never sees
    the earlier turns, and puts the agent's job and what the message depends on inside the six."""
    body = load_point("join-in-judge").generator_body
    kept = _NUMBER_WORDS[render.RECENT_TURNS_KEPT]
    paragraph = next(p for p in body.split("\n\n") if f"last {kept}" in p)
    flat = " ".join(paragraph.split())
    assert f"reads only the last {kept} of `priorTurns`" in flat
    assert "never sees the earliest" in flat
    assert f"must therefore sit inside the last {kept} `priorTurns`" in flat
    assert "shows the agent's job" in flat and "the new message depends on" in flat


def test_a_later_indented_sample_is_not_part_of_the_layout_block() -> None:
    section = "\nintro\n\n    A: <x>\n\n    B: <y>\n\nprose after the block\n\n    sample: later\n"
    assert layout_lines(section) == ["A: <x>", "", "B: <y>"]
    assert layout_lines("no block here\n") == []


def test_the_item_wrappers_filter_before_they_slice_and_cut_to_the_contract_lengths() -> None:
    """The wrappers are pinned by value, not by wiring (P26-m2): a slice before the filter would
    let a malformed tail row shrink the window, and `render_*_input` is what the vectors cover."""
    turns = [{"role": "user", "displayName": f"P{n}", "text": f"turn {n}"} for n in range(1, 7)]
    turns.append({"role": "user", "displayName": "P7", "text": ""})
    item = {**JOIN_IN_ITEM, "priorTurns": turns}
    rendered = render.render_join_in_item(item)
    assert rendered == render.render_join_in_input("Pastel", turns, item["newMessage"])
    assert [f"P{n}: turn {n}" in rendered for n in range(1, 8)] == [True] * 6 + [False]
    long_prior = {"speaker": "Shop Assistant", "text": "p" * 401}
    message = render.render_message_type_item({"text": "t" * 2001, "priorTurn": long_prior})
    assert message == ("PREVIOUS MESSAGE:\nShop Assistant: " + "p" * 400
                       + "\n\nMESSAGE:\n" + "t" * 2000)
    assert render.render_message_type_item({"text": "hi", "priorTurn": None}) == "MESSAGE:\nhi"


def test_the_join_in_agent_name_is_rendered_verbatim_never_stripped() -> None:
    """The contract's function does not strip, and no vector has padding (P26-n1)."""
    message = {"role": "user", "displayName": "Bo", "text": "hi"}
    assert render.render_join_in_input("  Pastel ", [], message) == (
        "AGENT:   Pastel \n\nNEW MESSAGE:\nBo: hi")
    item = {**JOIN_IN_ITEM, "agentName": " Pastel", "priorTurns": []}
    assert render.render_join_in_item(item).startswith("AGENT:  Pastel\n\n")


def test_the_join_in_new_message_speaker_is_the_display_name_before_the_author_id() -> None:
    """`render-join-in.jsonl` has no new message carrying both: only the fall-back cases."""
    both = {"role": "user", "displayName": "Bo", "authorId": "u9", "text": "hi"}
    rendered = render.render_join_in_input("Pastel", [], both)
    assert rendered == "AGENT: Pastel\n\nNEW MESSAGE:\nBo: hi"
