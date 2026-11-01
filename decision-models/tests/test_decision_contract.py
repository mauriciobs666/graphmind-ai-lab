"""The decision contract copies (typed-decision-models U5.2, U5.2b-hash; plans
`docs/plans/typed-decision-models4.md` §4.1 and `docs/plans/typed-decision-models5.md` §3.4 over
v1.3 §3.2, §3.2.2, §3.5).

model-bench owns the canonical golden vectors (`model-bench/tests/fixtures/decision-contract/`).
This suite asserts `decisionmodels.render` and `decisionmodels.hashing` against **byte-copies** of
two of them under `tests/fixtures/decision-contract/`:

* `render-guard.jsonl` — `render_guard_input(condition, understanding, recentTurns)`; expected
  strings produced by falkor-chat's real `_render_judge_user`;
* `input-hash.jsonl` — `input_hash(item, role)` for all three roles; digests computed by hand
  from v1.3 §3.5.

The guarded equality test asserts each copy byte-identical to model-bench's and skips only when
`model-bench/` is absent from the checkout (v1.3 §3.2.2's skip rule). Reading model-bench's files
here is not an FR-8 reach: `tests/test_fr8_isolation.py` scans `decisionmodels/` only, and the
package itself never reads them. A copy is refreshed by copying the canonical file, never by
regenerating it.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from decisionmodels import hashing, render

_COPIES = Path(__file__).resolve().parent / "fixtures" / "decision-contract"
_MODEL_BENCH = Path(__file__).resolve().parents[2] / "model-bench"
_CANONICAL = _MODEL_BENCH / "tests" / "fixtures" / "decision-contract"
_VECTOR_FILES = ("render-guard.jsonl", "input-hash.jsonl")


def _rows(name: str) -> list[dict[str, Any]]:
    with (_COPIES / name).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


_RENDER_ROWS = _rows("render-guard.jsonl")
_HASH_ROWS = _rows("input-hash.jsonl")


@pytest.mark.parametrize("name", _VECTOR_FILES)
def test_the_vector_copy_is_byte_identical_to_model_benchs(name: str) -> None:
    if not _MODEL_BENCH.is_dir():
        pytest.skip("model-bench/ is absent from this checkout (v1.3 §3.2.2 skip rule)")
    assert (_COPIES / name).read_bytes() == (_CANONICAL / name).read_bytes()


@pytest.mark.parametrize(
    ("name", "rows"),
    [("render-guard.jsonl", _RENDER_ROWS), ("input-hash.jsonl", _HASH_ROWS)],
)
def test_every_vector_line_is_a_parametrized_row(name: str, rows: list[dict[str, Any]]) -> None:
    """The loader drops no row: counted from the raw bytes, not from the loader."""
    lines = [line for line in (_COPIES / name).read_bytes().splitlines() if line.strip()]
    assert len(rows) == len(lines)


@pytest.mark.parametrize("row", _RENDER_ROWS, ids=[r["case"] for r in _RENDER_ROWS])
def test_render_guard_input_matches_render_guard_jsonl(row: dict[str, Any]) -> None:
    rendered = render.render_guard_input(row["condition"], row["understanding"], row["recentTurns"])
    assert rendered == row["expected"]


@pytest.mark.parametrize("row", _HASH_ROWS, ids=[r["case"] for r in _HASH_ROWS])
def test_input_hash_matches_input_hash_jsonl(row: dict[str, Any]) -> None:
    assert hashing.input_hash(row["item"], row["role"]) == row["expected"]


def test_input_hash_refuses_a_role_with_no_input_fields() -> None:
    with pytest.raises(ValueError, match="no input fields"):
        hashing.input_hash({"text": "hi"}, "no-such-role")


def _binds(node: ast.stmt, name: str) -> bool:
    """Whether a top-level statement binds `name`: an assignment target or an import alias."""
    if isinstance(node, ast.AnnAssign):
        return isinstance(node.target, ast.Name) and node.target.id == name
    if isinstance(node, ast.Assign):
        return any(
            isinstance(n, ast.Name) and n.id == name
            for target in node.targets
            for n in ast.walk(target)
        )
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return any((alias.asname or alias.name.split(".")[0]) == name for alias in node.names)
    return False


def _input_fields_literal(source: str) -> dict[str, tuple[str, ...]]:
    """`INPUT_FIELDS_BY_ROLE`'s literal in a `decisions.py` source, which must bind the name
    exactly once at top level, as `INPUT_FIELDS_BY_ROLE: … = MappingProxyType({…})`: a second
    binding would leave the literal stale (Pass 7 P7-F2)."""
    bindings = [n for n in ast.parse(source).body if _binds(n, "INPUT_FIELDS_BY_ROLE")]
    if len(bindings) != 1:
        raise AssertionError(f"INPUT_FIELDS_BY_ROLE is bound {len(bindings)} times, not once")
    node = bindings[0]
    if not (isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Call)):
        raise AssertionError("model-bench's decisions.py defines no INPUT_FIELDS_BY_ROLE literal")
    return ast.literal_eval(node.value.args[0])


def _sorted_rows(rows: Mapping[str, tuple[str, ...]]) -> dict[str, list[str]]:
    """Order-free but duplicate-sensitive: `canonical_input_json` sorts keys, so tuple order
    changes no hash (Pass 7 P7-F1)."""
    return {role: sorted(fields) for role, fields in rows.items()}


def test_input_fields_by_role_equals_model_benchs() -> None:
    """Read from model-bench's source, never imported (FR-8)."""
    if not _MODEL_BENCH.is_dir():
        pytest.skip("model-bench/ is absent from this checkout (v1.3 §3.2.2 skip rule)")
    source = (_MODEL_BENCH / "modelbench" / "decisions.py").read_text(encoding="utf-8")
    assert _sorted_rows(hashing.INPUT_FIELDS_BY_ROLE) == _sorted_rows(_input_fields_literal(source))


_ONE_BINDING = 'INPUT_FIELDS_BY_ROLE: Mapping = MappingProxyType({"r": ("a", "b")})\n'


@pytest.mark.parametrize(
    "second",
    [
        'INPUT_FIELDS_BY_ROLE = MappingProxyType({"r": ("a",)})\n',
        'INPUT_FIELDS_BY_ROLE: Mapping = MappingProxyType({"r": ("a",)})\n',
        "X, INPUT_FIELDS_BY_ROLE = 1, {}\n",
        "from elsewhere import INPUT_FIELDS_BY_ROLE\n",
        "from elsewhere import ROWS as INPUT_FIELDS_BY_ROLE\n",
    ],
)
def test_the_reader_refuses_a_second_binding(second: str) -> None:
    assert _input_fields_literal(_ONE_BINDING) == {"r": ("a", "b")}
    with pytest.raises(AssertionError, match="bound 2 times"):
        _input_fields_literal(_ONE_BINDING + second)


# --- turn normalization (hand-derived from the rule; no cross-component vector covers it) -------


def test_normalize_filters_invalid_rows_before_keeping_the_newest_six() -> None:
    rows = [{"text": f"t{i}", "displayName": f"u{i}"} for i in range(7)]
    rows += [{"text": ""}, {"text": None}, {"role": "user"}]
    turns = render.normalize_turns(rows)
    assert [t["text"] for t in turns] == ["t1", "t2", "t3", "t4", "t5", "t6"]


def test_normalize_speaker_role_and_text_cap() -> None:
    rows = [
        {"text": "a", "displayName": "Ana", "authorId": "u-1", "role": "assistant"},
        {"text": "b", "authorId": "u-2"},
        {"text": "x" * 401},
    ]
    assert render.normalize_turns(rows) == [
        {"speaker": "Ana", "role": "assistant", "text": "a"},
        {"speaker": "u-2", "role": "user", "text": "b"},
        {"speaker": "member", "role": "user", "text": "x" * 400},
    ]


def test_render_guard_item_normalizes_raw_turns_before_rendering() -> None:
    item = {
        "condition": "enough info?",
        "turns": [{"text": "hello", "displayName": "Ana", "role": "user", "createdAt": 1}],
        "path": "turns",
        "expected": True,
    }
    assert render.render_guard_item(item) == (
        "CONDITION: enough info?\n\nRECENT TURNS (context only):\nAna: hello"
    )


def test_render_guard_item_keeps_turn_order_through_normalization_and_the_cap() -> None:
    """Ten raw rows: one invalid, nine valid, so the last six valid (t3..t8) survive normalization;
    t4 falls back to `authorId`, t5 to `member`; t8's 450 chars are cut to 400. The state block is
    sized so the six turns total 6,008 chars and dropping the oldest (t3) leaves exactly 6,000.
    The expected string is written from v1.3 §3.2's rule, oldest turn first."""
    rows = [
        {"text": "t0", "displayName": "Ana"},
        {"text": "t1", "displayName": "Bo"},
        {"text": "", "displayName": "Ana"},
        {"text": "t2", "displayName": "Ana"},
        {"text": "t3", "displayName": "Ana", "role": "user"},
        {"text": "t4", "authorId": "u-7"},
        {"text": "t5", "role": "assistant"},
        {"text": "t6", "displayName": "Bo"},
        {"text": "t7", "displayName": "Ana"},
        {"text": "z" * 450, "displayName": "Bo", "authorId": "u-9"},
    ]
    item = {"condition": "the order is confirmed", "understanding": {"notes": "n" * 5464}}
    item["turns"] = rows
    expected = (
        "CONDITION: the order is confirmed\n\n"
        'CURRENT STATE:\n{\n  "notes": "' + "n" * 5464 + '"\n}\n\n'
        "RECENT TURNS (context only):\n"
        "u-7: t4\nmember: t5\nBo: t6\nAna: t7\nBo: " + "z" * 400
    )
    assert len(expected) == render.MAX_CHARS
    assert render.render_guard_item(item) == expected


def test_render_guard_item_with_no_understanding_and_no_turns_is_the_condition_alone() -> None:
    assert render.render_guard_item({"condition": "c"}) == "CONDITION: c"
