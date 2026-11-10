"""The request writer and its closed schemas (typed-decision-models U5.4; plan
`docs/plans/typed-decision-models4.md` §3.2, §5 U5.4; plan5 `typed-decision-models5.md` §3.4).

Expected values are written from the plans, not read from `decisionmodels.exchange`:
- the header's keys and `schema`, line 2's key per kind, the item and seed line keys (plan4 §3.2);
- `bodySha256`: sha256 over `json.dumps(body, sort_keys=True, separators=(",", ":"),
  ensure_ascii=False)` encoded UTF-8, recomputed here independently (`independent_body_sha256`);
- the page budget, 40,000 written characters, and `P_max` 10 (label) and 2 (generate);
- keys: 10 base32 characters;
- id-blind seed lines: no key ending in `Id` (plan5 §3.4).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from decisionmodels import exchange, render
from decisionmodels.points import load_point
from decisionmodels.seeds import draw_seeds

GUARD = load_point("guard-judge")

BUDGET = 40_000
P_MAX_LABEL = 10
P_MAX_GENERATE = 2
KEY_RE = re.compile(r"[a-z2-7]{10}")


def guard_item(index: int) -> dict[str, Any]:
    return {
        "condition": "the user has stated a budget for the request",
        "path": "turns",
        "understanding": {},
        "turns": [
            {"role": "user", "displayName": "Ana", "text": f"Quero um bolo para 20 ({index})"},
            {"role": "assistant", "displayName": "Assistant", "text": "Qual é o orçamento?"},
        ],
    }


PT_BR_PHRASE = (
    "O cliente já informou que precisa de orçamento para a reforma da cozinha, com instalação "
    "elétrica nova e acabamento em azulejo português"
)


def pt_br_item_at_cap(index: int) -> dict[str, Any]:
    """A guard `understanding` item whose rendering is exactly at the 6,000-character cap: the
    state block is pt-BR, so `render` escapes its accents (`\\u00e7`) and the request line escapes
    them again, the costliest realistic line."""
    return {
        "condition": "the user has stated a budget for the request",
        "path": "understanding",
        "understanding": {
            "request": f"Reforma da cozinha nº {index} com orçamento até R$ 15.000",
            "known": [f"{PT_BR_PHRASE} ({index}.{n})" for n in range(60)],
            "missing": [],
        },
        "turns": [],
    }


def label_request(items=None, *, tmp_path: Path, cap: int = 50, held=()) -> exchange.BuiltRequest:
    items = items if items is not None else [guard_item(i) for i in range(3)]
    return exchange.build_label_request(
        GUARD,
        [(f"it_{i:04d}", item) for i, item in enumerate(items)],
        exchange_dir=tmp_path,
        cap=cap,
        held_keys=set(held),
    )


def generation_request(*, tmp_path: Path, role: str = "guard-judge", count: int = 6, cap: int = 25):
    draw = draw_seeds(role, count, 7, design_stratum=None)
    return exchange.build_generation_request(
        load_point(role), draw.seeds, exchange_dir=tmp_path, cap=cap, held_keys=set()
    ), draw


def lines_of(text: str) -> list[dict[str, Any]]:
    assert text.endswith("\n")
    return [json.loads(line) for line in text[:-1].split("\n")]


def independent_body_sha256(text: str) -> str:
    """plan4 §3.2's body hash, recomputed from the file's lines without `exchange`."""
    header, line2, *entries = lines_of(text)
    if header["role"] == "label":
        body = {"rubric": line2.get("rubric"), "items": entries}
    else:
        body = {"instructions": line2.get("instructions"), "seeds": entries}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def relaid(text: str, change) -> str:
    """`text` with `change(lines)` applied and the header's `bodySha256` recomputed, so the only
    defect left is the one `change` planted."""
    lines = lines_of(text)
    change(lines)
    draft = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
    lines[0]["bodySha256"] = independent_body_sha256(draft)
    return "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)


# --- keys ----------------------------------------------------------------------------------------


def test_a_key_is_ten_base32_characters() -> None:
    assert KEY_RE.fullmatch(exchange.mint_key(set()))


def test_a_key_collision_is_redrawn(monkeypatch: pytest.MonkeyPatch) -> None:
    drawn = iter("aaaaaaaaaa" + "bbbbbbbbbb")
    monkeypatch.setattr(exchange.secrets, "choice", lambda alphabet: next(drawn))
    assert exchange.mint_key({"aaaaaaaaaa"}) == "bbbbbbbbbb"


def scripted_choice(monkeypatch: pytest.MonkeyPatch, *keys: str) -> None:
    """`exchange.secrets.choice` yields `keys`, then 16 characters for the opaque name."""
    drawn = iter("".join(keys) + "o" * 16)
    monkeypatch.setattr(exchange.secrets, "choice", lambda alphabet: next(drawn))


def test_a_request_key_avoids_a_held_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scripted_choice(monkeypatch, "aaaaaaaaaa", "bbbbbbbbbb")
    built = label_request([guard_item(0)], tmp_path=tmp_path, held={"aaaaaaaaaa"})
    assert list(built.keys) == ["bbbbbbbbbb"]


def test_request_keys_are_unique_within_a_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scripted_choice(monkeypatch, "aaaaaaaaaa", "aaaaaaaaaa", "bbbbbbbbbb")
    built = label_request([guard_item(0), guard_item(1)], tmp_path=tmp_path)
    assert list(built.keys) == ["aaaaaaaaaa", "bbbbbbbbbb"]


def test_the_key_map_follows_the_line_order(tmp_path: Path) -> None:
    built = label_request([guard_item(i) for i in range(40)], tmp_path=tmp_path)
    assert list(built.keys) == [line["key"] for line in lines_of(built.text)[2:]]


def test_a_rerun_gets_fresh_keys_and_a_new_opaque_name(tmp_path: Path) -> None:
    first = label_request(tmp_path=tmp_path)
    second = label_request(tmp_path=tmp_path, held=first.keys)
    assert first.opaque != second.opaque
    assert not set(first.keys) & set(second.keys)


# --- the label request ---------------------------------------------------------------------------


def test_a_label_request_is_json_lines_header_rubric_then_one_item_per_line(tmp_path: Path) -> None:
    items = [guard_item(i) for i in range(3)]
    built = label_request(items, tmp_path=tmp_path)
    header, line2, *entries = lines_of(built.text)
    assert set(header) == {"schema", "role", "point", "answerPath", "bodySha256", "pages"}
    assert header["schema"] == "dm-request/1"
    assert header["role"] == "label"
    assert header["point"] == "guard-judge"
    assert header["answerPath"] == str(tmp_path / "ans" / f"{built.opaque}.jsonl")
    assert line2 == {"rubric": GUARD.rubric}
    assert [set(entry) for entry in entries] == [{"key", "text"}] * 3
    assert [entry["text"] for entry in entries] == [render.render_guard_item(i) for i in items]
    assert [built.keys[entry["key"]] for entry in entries] == ["it_0000", "it_0001", "it_0002"]


def test_the_writers_label_request_passes_validate_request(tmp_path: Path) -> None:
    exchange.validate_request(label_request(tmp_path=tmp_path).text)


def test_write_request_writes_the_validated_text_under_req(tmp_path: Path) -> None:
    built = label_request(tmp_path=tmp_path)
    path = exchange.write_request(built)
    assert path == tmp_path / "req" / f"{built.opaque}.jsonl"
    assert path.read_text(encoding="utf-8") == built.text
    assert built.request_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_write_request_refuses_an_invalid_text_and_writes_nothing(tmp_path: Path) -> None:
    built = label_request(tmp_path=tmp_path)
    broken = dataclasses.replace(built, text=relaid(built.text, lambda ls: ls[2].update(x=1)))
    with pytest.raises(exchange.RequestRefused):
        exchange.write_request(broken)
    assert not (tmp_path / "req").exists() or not any((tmp_path / "req").iterdir())


def test_the_body_hash_matches_an_independent_recomputation(tmp_path: Path) -> None:
    built = label_request(tmp_path=tmp_path)
    assert lines_of(built.text)[0]["bodySha256"] == independent_body_sha256(built.text)


def test_the_body_hash_is_unchanged_by_the_file_layout(tmp_path: Path) -> None:
    items = [guard_item(i) for i in range(30)]
    built = label_request(items, tmp_path=tmp_path)
    keys = [line["key"] for line in lines_of(built.text)[2:]]

    def reordered_keys_and_smaller_pages(lines: list[dict[str, Any]]) -> None:
        for index in range(1, len(lines)):
            lines[index] = dict(reversed(list(lines[index].items())))
        lines[0]["pages"] = [[1, 1]] + [[n, 1] for n in range(2, len(lines) + 1)]

    lines = lines_of(built.text)
    reordered_keys_and_smaller_pages(lines)
    relaid_text = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
    assert relaid_text != built.text
    assert independent_body_sha256(relaid_text) == lines_of(built.text)[0]["bodySha256"]
    assert [line["key"] for line in lines_of(relaid_text)[2:]] == keys


def test_a_label_request_for_a_role_without_a_renderer_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="renderer"):
        exchange.build_label_request(
            dataclasses.replace(load_point("join-in-judge"), role="no-such-role"),
            [("it_1", {})], exchange_dir=tmp_path, cap=50, held_keys=set(),
        )


@pytest.mark.parametrize("role", ["join-in-judge", "message-type-classifier"])
def test_every_point_has_a_label_renderer(role: str) -> None:
    assert role in exchange.RENDERERS


# --- positive controls: one planted key per line kind (plan4 §5 U5.4) ----------------------------


@pytest.mark.parametrize(
    ("line_index", "planted", "kind"),
    [(0, "split", "header"), (1, "seedId", "rubric"), (2, "intent", "item")],
)
def test_a_planted_key_is_refused_on_each_label_line_kind(
    tmp_path: Path, line_index: int, planted: str, kind: str
) -> None:
    text = relaid(
        label_request(tmp_path=tmp_path).text, lambda ls: ls[line_index].update({planted: "x"})
    )
    errors = exchange.request_errors(text)
    assert errors == [f"line {line_index + 1} ({kind}): unexpected key {planted!r}"]
    with pytest.raises(exchange.RequestRefused):
        exchange.validate_request(text)


@pytest.mark.parametrize(
    ("line_index", "missing", "kind"),
    [(0, "point", "header"), (2, "text", "item")],
)
def test_a_missing_key_is_refused(tmp_path: Path, line_index: int, missing: str, kind: str) -> None:
    text = relaid(label_request(tmp_path=tmp_path).text, lambda ls: ls[line_index].pop(missing))
    assert f"line {line_index + 1} ({kind}): missing key {missing!r}" in exchange.request_errors(
        text
    )


def test_a_duplicate_or_malformed_request_key_is_refused(tmp_path: Path) -> None:
    text = label_request(tmp_path=tmp_path).text
    duplicate = relaid(text, lambda ls: ls[3].update(key=ls[2]["key"]))
    assert exchange.request_errors(duplicate) == [
        f"line 4 (item): duplicate key {lines_of(text)[2]['key']!r}"
    ]
    upper = relaid(text, lambda ls: ls[2].update(key="ABCDEFGHIJ"))
    assert exchange.request_errors(upper) == [
        "line 3 (item): key 'ABCDEFGHIJ' is not 10 base32 characters"
    ]


def test_two_key_less_lines_are_two_missing_keys_not_a_duplicate(tmp_path: Path) -> None:
    text = relaid(label_request(tmp_path=tmp_path).text,
                  lambda ls: (ls[2].pop("key"), ls[3].pop("key")))
    assert exchange.request_errors(text) == [
        "line 3 (item): missing key 'key'",
        "line 4 (item): missing key 'key'",
    ]


def test_a_header_missing_its_hash_reports_it_once(tmp_path: Path) -> None:
    text = relaid(label_request(tmp_path=tmp_path).text, lambda ls: None)
    lines = lines_of(text)
    del lines[0]["bodySha256"]
    stripped = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
    assert exchange.request_errors(stripped) == ["line 1 (header): missing key 'bodySha256'"]


def test_a_wrong_header_hash_is_refused(tmp_path: Path) -> None:
    text = label_request(tmp_path=tmp_path).text
    lines = lines_of(text)
    lines[0]["bodySha256"] = "0" * 64
    tampered = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
    assert exchange.request_errors(tampered) == [
        "line 1 (header): bodySha256 does not match the body"
    ]


def test_a_generation_line_2_in_a_label_request_is_refused(tmp_path: Path) -> None:
    def swap(lines: list[dict[str, Any]]) -> None:
        lines[1] = {"instructions": lines[1]["rubric"]}

    errors = exchange.request_errors(relaid(label_request(tmp_path=tmp_path).text, swap))
    assert "line 2 (rubric): unexpected key 'instructions'" in errors


# --- paging --------------------------------------------------------------------------------------


def written(text: str, page: list[int]) -> int:
    """The written characters of one `[offset, limit]` page: its lines, newlines included."""
    offset, limit = page
    return sum(len(line) + 1 for line in text[:-1].split("\n")[offset - 1 : offset - 1 + limit])


def assert_pages_cover_every_line_once(text: str, pages: list[list[int]]) -> None:
    covered = [n for offset, limit in pages for n in range(offset, offset + limit)]
    assert covered == list(range(1, text.count("\n") + 1))


def test_the_first_page_is_the_header_alone(tmp_path: Path) -> None:
    assert lines_of(label_request(tmp_path=tmp_path).text)[0]["pages"][0] == [1, 1]


def test_fifty_pt_br_items_at_the_render_cap_page_within_the_budget(tmp_path: Path) -> None:
    items = [pt_br_item_at_cap(i) for i in range(50)]
    assert {len(render.render_guard_item(item)) for item in items} == {render.MAX_CHARS}
    built = label_request(items, tmp_path=tmp_path)
    pages = lines_of(built.text)[0]["pages"]
    assert built.overflow == ()
    assert len(lines_of(built.text)) == 52
    assert len(pages) <= P_MAX_LABEL
    assert all(written(built.text, page) <= BUDGET for page in pages)
    assert_pages_cover_every_line_once(built.text, pages)
    exchange.validate_request(built.text)


def test_a_label_batch_that_would_exceed_p_max_pages_is_cut_there(tmp_path: Path) -> None:
    items = [pt_br_item_at_cap(i) for i in range(70)]
    built = label_request(items, tmp_path=tmp_path, cap=70)
    lines = lines_of(built.text)
    pages = lines[0]["pages"]
    assert len(pages) == P_MAX_LABEL
    assert_pages_cover_every_line_once(built.text, pages)
    kept = len(lines) - 2
    assert 50 <= kept < 70
    assert [ref for ref, _ in built.overflow] == [f"it_{i:04d}" for i in range(kept, 70)]
    # One more item would not have fit on the last page.
    next_entry = {"key": "k" * 10, "text": render.render_guard_item(items[kept])}
    next_line = len(json.dumps(next_entry, ensure_ascii=False, separators=(",", ":"))) + 1
    assert written(built.text, pages[-1]) + next_line > BUDGET


def test_items_past_the_cap_overflow_to_the_next_batch(tmp_path: Path) -> None:
    built = label_request([guard_item(i) for i in range(5)], tmp_path=tmp_path, cap=3)
    assert len(lines_of(built.text)) == 2 + 3
    assert [ref for ref, _ in built.overflow] == ["it_0003", "it_0004"]


def test_a_page_holds_exactly_the_budget_and_not_one_character_more() -> None:
    # Two lines of 20,000 written characters each (19,999 + newline) fill one page exactly.
    assert exchange.cut_pages([19_999, 19_999, 5], budget=BUDGET, max_pages=10) == (
        [[1, 1], [2, 2], [4, 1]],
        3,
    )
    assert exchange.cut_pages([19_999, 20_000, 5], budget=BUDGET, max_pages=10) == (
        [[1, 1], [2, 1], [3, 2]],
        3,
    )


def test_cut_pages_stops_at_max_pages_and_reports_how_many_lines_fit() -> None:
    assert exchange.cut_pages([30_000] * 5, budget=BUDGET, max_pages=3) == (
        [[1, 1], [2, 1], [3, 1]],
        2,
    )


def test_a_line_of_exactly_the_budget_fits_a_page_alone() -> None:
    assert exchange.cut_pages([BUDGET - 1, 3], budget=BUDGET, max_pages=10) == (
        [[1, 1], [2, 1], [3, 1]],
        2,
    )


def test_cut_pages_refuses_a_line_longer_than_the_budget() -> None:
    with pytest.raises(ValueError, match="budget"):
        exchange.cut_pages([BUDGET], budget=BUDGET, max_pages=10)


def test_a_page_over_the_budget_is_refused(tmp_path: Path) -> None:
    built = label_request([pt_br_item_at_cap(i) for i in range(8)], tmp_path=tmp_path)

    def one_page(lines: list[dict[str, Any]]) -> None:
        lines[0]["pages"] = [[1, 1], [2, len(lines) - 1]]

    text = relaid(built.text, one_page)
    assert written(text, [2, 9]) > BUDGET
    assert exchange.request_errors(text) == [
        f"line 1 (header): page [2, 9] holds {written(text, [2, 9])} characters, over {BUDGET}"
    ]


def page_of_exactly(text: str, target: int) -> str:
    """`text` laid out as the header page plus one page of lines 2…, the last item's text padded
    so that page holds exactly `target` written characters."""

    def one_page(lines: list[dict[str, Any]]) -> None:
        lines[0]["pages"] = [[1, 1], [2, len(lines) - 1]]

    laid = relaid(text, one_page)
    pad = target - written(laid, [2, laid.count("\n") - 1])
    assert pad >= 0
    return relaid(laid, lambda ls: ls[-1].update(text=ls[-1]["text"] + "a" * pad))


def test_a_page_of_exactly_the_budget_is_valid_and_one_more_is_not(tmp_path: Path) -> None:
    text = label_request([guard_item(i) for i in range(3)], tmp_path=tmp_path).text
    at_budget = page_of_exactly(text, BUDGET)
    assert written(at_budget, [2, 4]) == BUDGET
    assert exchange.request_errors(at_budget) == []
    over = page_of_exactly(text, BUDGET + 1)
    assert exchange.request_errors(over) == [
        f"line 1 (header): page [2, 4] holds {BUDGET + 1} characters, over {BUDGET}"
    ]


@pytest.mark.parametrize(
    "pages",
    [[[1, 1], [0, 0], [2, 4]], [[1, 1], [2, 4], [6, 0]], [[1, 1], [2, 4], [3, -1]]],
)
def test_a_page_with_a_limit_below_one_is_refused(tmp_path: Path, pages: list) -> None:
    built = label_request([guard_item(i) for i in range(3)], tmp_path=tmp_path)
    text = relaid(built.text, lambda ls: ls[0].update(pages=pages))
    assert "line 1 (header): a page's limit is below 1" in exchange.request_errors(text)


def test_a_request_with_no_entries_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no items"):
        label_request([], tmp_path=tmp_path)
    with pytest.raises(ValueError, match="no seeds"):
        exchange.build_generation_request(
            GUARD, (), exchange_dir=tmp_path, cap=25, held_keys=set()
        )
    built = label_request(tmp_path=tmp_path)
    no_entries = relaid(built.text, lambda ls: (ls.__delitem__(slice(2, None)),
                                                ls[0].update(pages=[[1, 1], [2, 1]])))
    assert exchange.request_errors(no_entries) == ["a request holds one entry at least"]


def test_a_repeated_name_in_a_request_line_is_refused(tmp_path: Path) -> None:
    text = label_request(tmp_path=tmp_path).text
    first, rest = text.split("\n", 1)
    doubled = first[:-1] + ',"split":"eval","split":"eval"}\n' + rest
    assert "line 1: repeated name 'split'" in exchange.request_errors(doubled)


@pytest.mark.parametrize(
    ("planted", "error"),
    [
        ("[" * 100_000 + "]" * 100_000, "line 3: nested too deeply"),
        ("1" * 5_000, "line 3: not JSON"),
    ],
)
def test_an_unparseable_request_line_is_refused_not_raised(
    tmp_path: Path, planted: str, error: str
) -> None:
    """A deep nesting raises `RecursionError` and a 5,000-digit integer a plain `ValueError`
    (Python's integer-string limit): both are refusals (review Pass 6 P6-F3)."""
    lines = label_request(tmp_path=tmp_path).text.split("\n")
    lines[2] = lines[2][:-1] + ',"x":' + planted + "}"
    errors = exchange.request_errors("\n".join(lines))
    assert len(errors) == 1 and errors[0].startswith(error), errors


def test_a_page_list_that_skips_or_repeats_a_line_is_refused(tmp_path: Path) -> None:
    built = label_request([guard_item(i) for i in range(3)], tmp_path=tmp_path)
    for pages in ([[1, 1], [2, 2]], [[1, 1], [2, 4], [5, 1]], [[1, 2], [3, 3]]):
        text = relaid(built.text, lambda ls, p=pages: ls[0].update(pages=p))
        assert any("pages" in error for error in exchange.request_errors(text)), pages


def test_more_pages_than_p_max_is_refused(tmp_path: Path) -> None:
    built = label_request([guard_item(i) for i in range(12)], tmp_path=tmp_path)
    one_per_line = [[1, 1]] + [[n, 1] for n in range(2, 15)]
    text = relaid(built.text, lambda ls: ls[0].update(pages=one_per_line))
    assert exchange.request_errors(text) == [
        f"line 1 (header): {len(one_per_line)} pages, over P_max {P_MAX_LABEL}"
    ]


def test_line_breaks_other_than_newline_are_written_escaped(tmp_path: Path) -> None:
    item = guard_item(0)
    item["turns"][0]["text"] = "linha separada e\x85fim"
    built = label_request([item], tmp_path=tmp_path)
    assert not {" ", " ", "\x85"} & set(built.text)
    assert lines_of(built.text)[2]["text"] == render.render_guard_item(item)
    exchange.validate_request(built.text)


def test_a_raw_unicode_line_separator_is_refused(tmp_path: Path) -> None:
    text = label_request(tmp_path=tmp_path).text
    raw = relaid(text, lambda ls: ls[2].update(text=ls[2]["text"] + " "))
    raw = raw.replace("\\u2028", " ")
    assert " " in raw
    assert any("line separator" in error for error in exchange.request_errors(raw))


# --- the generation request (plan5 §3.4: id-blind seed lines) ------------------------------------


@pytest.mark.parametrize("role", ["guard-judge", "join-in-judge", "message-type-classifier"])
def test_a_seed_line_from_the_writer_holds_no_id_key(tmp_path: Path, role: str) -> None:
    built, draw = generation_request(tmp_path=tmp_path, role=role)
    assert any(key.endswith("Id") for seed in draw.seeds for key in seed.axes)
    entries = lines_of(built.text)[2:]
    assert entries
    for entry in entries:
        assert not [key for key in entry if key.endswith("Id")]


def test_a_seed_line_is_the_key_plus_the_seeds_text_axes(tmp_path: Path) -> None:
    built, draw = generation_request(tmp_path=tmp_path)
    header, line2, *entries = lines_of(built.text)
    assert header["role"] == "generate"
    assert line2 == {"instructions": GUARD.generator_instructions}
    for entry in entries:
        seed = built.keys[entry["key"]]
        assert seed in draw.seeds
        text_axes = {k: v for k, v in seed.axes.items() if not k.endswith("Id")}
        assert entry == {"key": entry["key"], **text_axes}


def test_the_store_keeps_the_id_axes_by_request_key(tmp_path: Path) -> None:
    built, _ = generation_request(tmp_path=tmp_path)
    for seed in built.keys.values():
        assert {"conditionId", "domainId"} <= set(seed.axes)


def test_the_writers_generation_request_passes_validate_request(tmp_path: Path) -> None:
    built, _ = generation_request(tmp_path=tmp_path, count=25)
    exchange.validate_request(built.text)
    assert len(lines_of(built.text)[0]["pages"]) <= P_MAX_GENERATE


def test_a_planted_domain_id_on_a_generation_line_is_refused(tmp_path: Path) -> None:
    built, _ = generation_request(tmp_path=tmp_path)
    text = relaid(built.text, lambda ls: ls[2].update(domainId="dom_07"))
    assert exchange.request_errors(text) == [
        "line 3 (seed): unexpected key 'domainId' (no *Id key reaches the generator)"
    ]


@pytest.mark.parametrize(("line_index", "kind"), [(0, "header"), (1, "instructions"), (2, "seed")])
def test_a_planted_seed_id_is_refused_on_each_generation_line_kind(
    tmp_path: Path, line_index: int, kind: str
) -> None:
    built, _ = generation_request(tmp_path=tmp_path)
    text = relaid(built.text, lambda ls: ls[line_index].update(seedId="sd_000000000001"))
    assert exchange.request_errors(text) == [
        f"line {line_index + 1} ({kind}): unexpected key 'seedId'"
        " (no *Id key reaches the generator)"
    ]


def test_an_id_axis_is_refused_on_a_seed_line_even_when_the_point_declares_it(
    tmp_path: Path,
) -> None:
    """plan5 §3.4: "whatever `points/` declares" — a point declaring an axis named `regionId` does
    not make it sendable, so the `*Id` rule is not just the closed key set."""
    built, _ = generation_request(tmp_path=tmp_path)
    design = json.loads(json.dumps(GUARD.definition["design"]))
    design["balanced"].append({"axis": "regionId", "values": [{"value": "north", "weight": 1}]})
    point = dataclasses.replace(GUARD, definition={**GUARD.definition, "design": design})
    assert "regionId" in exchange.seed_axis_names(point)
    text = relaid(built.text, lambda ls: ls[2].update(regionId="north"))
    assert exchange.request_errors(text, point=point) == [
        "line 3 (seed): unexpected key 'regionId' (no *Id key reaches the generator)"
    ]


def test_a_seed_line_missing_a_text_axis_is_refused(tmp_path: Path) -> None:
    built, _ = generation_request(tmp_path=tmp_path)
    text = relaid(built.text, lambda ls: ls[2].pop("domain"))
    assert exchange.request_errors(text) == ["line 3 (seed): missing key 'domain'"]


def test_a_generation_batch_is_cut_at_p_max_two_pages(tmp_path: Path) -> None:
    built, _ = generation_request(tmp_path=tmp_path, count=400, cap=400)
    pages = lines_of(built.text)[0]["pages"]
    assert len(pages) == P_MAX_GENERATE
    assert built.overflow
    assert_pages_cover_every_line_once(built.text, pages)
    assert written(built.text, pages[1]) <= BUDGET
