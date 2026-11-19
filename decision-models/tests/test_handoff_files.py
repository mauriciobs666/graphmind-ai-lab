"""U6f.3b (`docs/plans/typed-decision-models4.md` §3.11): `fit/<role>/handoff.md` is a byte-copy of
the system prompt of the point's model-bench pack, and `fit/` text carries no forbidden fragment.

The pack prompt path is read from the pack's `pack.json` (`prompt.systemPrompt`), never hard-coded.
The equality tests skip only when `model-bench/` is absent from the checkout (v1.3 §3.2.2); a
present `model-bench/` with a missing pack fails. Tests may read model-bench; `decisionmodels/` may
not (FR-8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_fr8_isolation import FORBIDDEN_FRAGMENTS

from decisionmodels.points import ROLES

COMPONENT_DIR = Path(__file__).resolve().parents[1]
FIT_DIR = COMPONENT_DIR / "fit"
_MODEL_BENCH = COMPONENT_DIR.parent / "model-bench"


def _handoff(role: str, fit_dir: Path = FIT_DIR) -> Path:
    return fit_dir / role / "handoff.md"


def _pack_prompt(role: str, model_bench: Path = _MODEL_BENCH) -> Path:
    """The pack's system-prompt file, named by its `pack.json`. Skips only when `model-bench/`
    is absent; a present one without the pack or file fails."""
    if not model_bench.is_dir():
        pytest.skip("model-bench/ is absent from this checkout (v1.3 §3.2.2 skip rule)")
    pack_json = model_bench / "packs" / f"{role}-synthetic" / "pack.json"
    assert pack_json.is_file(), f"model-bench/ is present but {pack_json} is not"
    name = json.loads(pack_json.read_text(encoding="utf-8"))["prompt"]["systemPrompt"]
    prompt = pack_json.parent / name
    assert prompt.is_file(), f"model-bench/ is present but {prompt} is not"
    return prompt


def _forbidden_in(fit_dir: Path) -> list[tuple[str, str]]:
    return [
        (str(path.relative_to(fit_dir)), name)
        for path in sorted(fit_dir.rglob("handoff.md"))
        for name, pattern in FORBIDDEN_FRAGMENTS.items()
        if pattern.search(path.read_text(encoding="utf-8").lower())
    ]


def _copy_fit(tmp_path: Path) -> Path:
    fit = tmp_path / "fit"
    for role in ROLES:
        _handoff(role, fit).parent.mkdir(parents=True)
        _handoff(role, fit).write_bytes(_handoff(role).read_bytes())
    return fit


def test_every_role_has_a_handoff_file() -> None:
    assert len(ROLES) == 3
    for role in ROLES:
        assert _handoff(role).is_file(), role
        assert _handoff(role).read_bytes().strip()


def _equals_pack(role: str, handoff: Path) -> bool:
    return handoff.read_bytes() == _pack_prompt(role).read_bytes()


@pytest.mark.parametrize("role", ROLES)
def test_handoff_equals_its_pack_prompt_byte_for_byte(role: str) -> None:
    assert _equals_pack(role, _handoff(role))


@pytest.mark.parametrize("role", ROLES)
def test_a_one_byte_edit_of_the_handoff_is_detected(role: str, tmp_path: Path) -> None:
    edited = tmp_path / "handoff.md"
    original = _handoff(role).read_bytes()
    edited.write_bytes(original)
    assert _equals_pack(role, edited)
    edited.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    assert not _equals_pack(role, edited)
    edited.write_bytes(original + b" ")
    assert not _equals_pack(role, edited)


def test_the_pack_prompt_path_comes_from_pack_json(tmp_path: Path) -> None:
    pack = tmp_path / "packs" / "guard-judge-synthetic"
    (pack / "elsewhere").mkdir(parents=True)
    (pack / "pack.json").write_text(
        json.dumps({"prompt": {"systemPrompt": "elsewhere/p.md"}}), encoding="utf-8"
    )
    (pack / "elsewhere" / "p.md").write_text("x", encoding="utf-8")
    assert _pack_prompt("guard-judge", tmp_path) == pack / "elsewhere" / "p.md"


def test_a_present_model_bench_without_the_pack_fails_rather_than_skips(tmp_path: Path) -> None:
    try:
        _pack_prompt("guard-judge", tmp_path)
    except pytest.skip.Exception:
        pytest.fail("a present model-bench/ without the pack skipped")
    except AssertionError as error:
        assert "is present but" in str(error)
    else:
        pytest.fail("a present model-bench/ without the pack passed")


def test_an_absent_model_bench_skips(tmp_path: Path) -> None:
    with pytest.raises(pytest.skip.Exception):
        _pack_prompt("guard-judge", tmp_path / "missing")


def test_no_handoff_carries_a_forbidden_fragment() -> None:
    assert _forbidden_in(FIT_DIR) == []


@pytest.mark.parametrize("fragment", sorted(FORBIDDEN_FRAGMENTS))
def test_a_planted_forbidden_fragment_in_a_handoff_is_refused(
    fragment: str, tmp_path: Path
) -> None:
    fit = _copy_fit(tmp_path)
    assert _forbidden_in(fit) == []
    target = _handoff(ROLES[0], fit)
    planted = target.read_text(encoding="utf-8") + f"\n{fragment.upper()}\n"
    target.write_text(planted, encoding="utf-8")
    assert _forbidden_in(fit) == [(f"{ROLES[0]}/handoff.md", fragment)]
