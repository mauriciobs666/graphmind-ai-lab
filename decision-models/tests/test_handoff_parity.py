"""U6f.3c (`docs/plans/typed-decision-models4.md` §3.11): the hand-off call's parity with the
model-bench pack and with the guard's served prompt.

Constants and prompt are compared with the pack / the sibling checkout's source. Every comparison
skips only when the sibling checkout is absent (v1.3 §3.2.2 skip rule); a present checkout with
the thing missing fails. Tests may read model-bench and falkor-chat; `decisionmodels/` may not
(FR-8)."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest
from test_handoff_files import _handoff, _pack_prompt

from decisionmodels import handoff
from decisionmodels.points import ROLES

_REPO = Path(__file__).resolve().parents[2]
_MODEL_BENCH = _REPO / "model-bench"
_APP = _REPO / "falkor-chat" / "server" / "falkorchat" / "app.py"
_RUNTIME_NAME = "_JUDGE_SYSTEM_PROMPT"


def _pack_json(role: str) -> dict:
    prompt = _pack_prompt(role)  # skips when model-bench/ is absent, fails on a missing pack
    return json.loads((prompt.parents[1] / "pack.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("role", ROLES)
def test_the_sampling_constants_equal_the_packs(role: str) -> None:
    prompt = _pack_json(role)["prompt"]
    assert handoff.HANDOFF_TEMPERATURE == prompt["temperature"]
    assert handoff.HANDOFF_MAX_TOKENS == prompt["maxTokens"]


def test_the_chat_timeout_equals_the_benchs_decision_point_timeout() -> None:
    source = _MODEL_BENCH / "modelbench" / "decisions.py"
    if not _MODEL_BENCH.is_dir():
        pytest.skip("model-bench/ is absent from this checkout (v1.3 §3.2.2 skip rule)")
    assert source.is_file(), f"model-bench/ is present but {source} is not"
    match = re.search(
        r"^DECISION_POINT_CHAT_TIMEOUT_SECONDS\s*=\s*([0-9.]+)\s*$",
        source.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match, "DECISION_POINT_CHAT_TIMEOUT_SECONDS assignment not found"
    assert handoff.HANDOFF_CHAT_TIMEOUT_SECONDS == float(match.group(1))


def test_the_constants_are_floats_and_ints_as_sent() -> None:
    assert isinstance(handoff.HANDOFF_TEMPERATURE, float)
    assert isinstance(handoff.HANDOFF_MAX_TOKENS, int)
    assert isinstance(handoff.HANDOFF_CHAT_TIMEOUT_SECONDS, float)


@pytest.mark.parametrize("role", ROLES)
def test_handoff_messages_is_system_prompt_then_the_rendering(role: str) -> None:
    rendered = "CONDITION: x\nRECENT TURNS:\n[a]: hi"
    assert handoff.handoff_messages(role, rendered) == [
        {"role": "system", "content": _handoff(role).read_text(encoding="utf-8")},
        {"role": "user", "content": rendered},
    ]


def test_handoff_messages_system_differs_per_role() -> None:
    systems = {handoff.handoff_messages(role, "x")[0]["content"] for role in ROLES}
    assert len(systems) == len(ROLES)


def test_handoff_messages_rejects_an_unknown_role() -> None:
    with pytest.raises(ValueError, match="unknown role"):
        handoff.handoff_messages("nope", "x")


def _runtime_prompt(app: Path) -> str:
    """`_JUDGE_SYSTEM_PROMPT` from `app.py`'s source, by `ast`. Skips only when `app.py`'s
    checkout is absent; an `app.py` without a module-level assignment of the name fails."""
    if not app.is_file():
        pytest.skip("falkor-chat/ is absent from this checkout (v1.3 §3.2.2 skip rule)")
    tree = ast.parse(app.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == _RUNTIME_NAME for t in node.targets
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, str), f"{_RUNTIME_NAME} is not a string literal"
            return value
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == _RUNTIME_NAME
            and node.value is not None
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, str), f"{_RUNTIME_NAME} is not a string literal"
            return value
    pytest.fail(f"{app} exists but {_RUNTIME_NAME} is not a module-level assignment")


def test_the_guard_handoff_equals_the_served_judge_prompt() -> None:
    pack_prompt = _pack_prompt("guard-judge").read_text(encoding="utf-8")
    assert _runtime_prompt(_APP) == pack_prompt.strip()


def test_a_one_word_edit_of_the_served_prompt_is_detected(tmp_path: Path) -> None:
    app = tmp_path / "app.py"
    pack_prompt = _pack_prompt("guard-judge").read_text(encoding="utf-8").strip()
    app.write_text(f"{_RUNTIME_NAME} = {pack_prompt!r}\n", encoding="utf-8")
    assert _runtime_prompt(app) == pack_prompt
    edited = pack_prompt.replace("strict", "lax")
    app.write_text(f"{_RUNTIME_NAME} = {edited!r}\n", encoding="utf-8")
    assert _runtime_prompt(app) != pack_prompt


def test_implicit_concatenation_and_a_trailing_newline_are_read(tmp_path: Path) -> None:
    app = tmp_path / "app.py"
    app.write_text(f'{_RUNTIME_NAME} = (\n    "a "\n    "b"\n)\n', encoding="utf-8")
    assert _runtime_prompt(app) == "a b"


@pytest.mark.parametrize(
    "source",
    [
        '_RENAMED = "x"\n',
        f'def f():\n    {_RUNTIME_NAME} = "x"\n',
        f'class C:\n    {_RUNTIME_NAME} = "x"\n',
        f"{_RUNTIME_NAME}: str\n",
        "",
    ],
    ids=["renamed", "in-function", "in-class", "bare-annotation", "empty"],
)
def test_a_present_app_without_a_module_level_assignment_fails_not_skips(
    source: str, tmp_path: Path
) -> None:
    app = tmp_path / "app.py"
    app.write_text(source, encoding="utf-8")
    try:
        _runtime_prompt(app)
    except pytest.skip.Exception:
        pytest.fail("a present app.py without the assignment skipped")
    except pytest.fail.Exception as error:
        assert "module-level assignment" in str(error)
    else:
        pytest.fail("a present app.py without the assignment passed")


def test_an_absent_checkout_skips(tmp_path: Path) -> None:
    with pytest.raises(pytest.skip.Exception):
        _runtime_prompt(tmp_path / "missing" / "app.py")
