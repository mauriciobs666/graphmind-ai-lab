"""FR-23 — the default (network-free) `pytest -q` suite must not depend on `falkor-chat/` being
present on disk (`AGENTS.md`'s "Standalone" hard rule). Three tests genuinely need a real
`falkor-chat/` source file to live-verify against (`test_refresh_golden.py`,
`test_scoring_extraction.py`); this module pins that they carry `conftest.py`'s
`requires_falkor_chat` skip guard, on a precondition distinct from the existing `live` marker
(that one means "needs a reachable LM Studio" — a different axis entirely, `AGENTS.md`
Conventions).

Two things are pinned, deliberately kept apart (plan discipline this repo applies elsewhere,
Conventions' "guard's reach" rule):

1. `falkor_chat_present()` itself: a pure function, tested both directions with `tmp_path` — no
   real `falkor-chat/` involved.
2. The three named tests actually carry the guard: import each module and inspect the target
   function's `pytestmark` for a `skipif` mark. This is the reproduction test for the shipped
   gap — before the fix, none of the three carried any guard at all, so the default suite reported
   `FileNotFoundError`s instead of clean skips when `falkor-chat/` was absent (live-confirmed this
   session: renaming `falkor-chat/` away made exactly these three fail, `3 failed, 1703 passed`).
"""

from __future__ import annotations

from pathlib import Path

import test_refresh_golden as _refresh_golden_tests
import test_scoring_extraction as _scoring_extraction_tests
from conftest import falkor_chat_present, requires_falkor_chat

# (module, function-name) for every test that reads a real `falkor-chat/` source file directly —
# the exact three named in this fix's own defect report.
_FALKOR_CHAT_DEPENDENT_TESTS = (
    (_refresh_golden_tests, "test_read_catalog_literal_handles_the_real_seed_catalog_script"),
    (_refresh_golden_tests, "test_read_schema_literal_handles_the_real_querygen_module"),
    (
        _scoring_extraction_tests,
        "test_prompts_querygen_md_matches_the_live_falkorchat_source_byte_for_byte",
    ),
)


def _skip_marks(func) -> list:
    return [mark for mark in getattr(func, "pytestmark", []) if mark.name == "skipif"]


def test_falkor_chat_present_is_true_for_a_directory_that_exists(tmp_path: Path) -> None:
    root = tmp_path / "falkor-chat"
    root.mkdir()
    assert falkor_chat_present(root) is True


def test_falkor_chat_present_is_false_for_a_missing_directory(tmp_path: Path) -> None:
    root = tmp_path / "falkor-chat"  # never created
    assert falkor_chat_present(root) is False


def test_falkor_chat_dependent_tests_each_carry_the_skip_guard() -> None:
    """The reproduction test for the shipped gap: each of the three tests that reads a real
    `falkor-chat/` file must carry a `skipif` mark whose reason names `falkor-chat` — distinct
    from (never reusing) the `live` marker, which gates LM-Studio reachability instead."""
    for module, name in _FALKOR_CHAT_DEPENDENT_TESTS:
        func = getattr(module, name)
        marks = _skip_marks(func)
        assert marks, f"{module.__name__}.{name} carries no skipif guard"
        assert any("falkor-chat" in str(mark.kwargs.get("reason", "")) for mark in marks), (
            f"{module.__name__}.{name}'s skipif guard doesn't name falkor-chat as the precondition"
        )
        live_marks = [
            mark
            for mark in getattr(func, "pytestmark", [])
            if mark.name == "live"
        ]
        assert not live_marks, (
            f"{module.__name__}.{name} must not reuse the `live` marker for the falkor-chat "
            "precondition — that marker means LM Studio reachability, a different axis"
        )


def test_requires_falkor_chat_marker_is_a_skipif_keyed_on_falkor_chat_presence() -> None:
    """`requires_falkor_chat` itself must be `skipif`-shaped (not the `live` marker reused) and
    its condition must match the real directory's current presence."""
    assert requires_falkor_chat.name == "skipif"
    (condition,) = requires_falkor_chat.args
    assert condition is (not falkor_chat_present())
