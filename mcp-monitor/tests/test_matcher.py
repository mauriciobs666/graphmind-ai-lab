"""Unit tests for `mcp_monitor.watch`'s regex/dedupe logic, in isolation from
any transport (no connection, no subprocess, no asyncio) — plan §11 tier 1.
"""

from __future__ import annotations

import re

from mcp_monitor.watch import filter_new, find_matches


def test_find_matches_returns_every_non_overlapping_match():
    pattern = re.compile(r"@mcp-monitor\b")
    text = "hey @mcp-monitor please look, also @mcp-monitor again"
    assert find_matches(pattern, text) == ["@mcp-monitor", "@mcp-monitor"]


def test_find_matches_empty_when_no_match():
    pattern = re.compile(r"@mcp-monitor\b")
    assert find_matches(pattern, "nothing interesting here") == []


def test_find_matches_supports_capture_groups_but_returns_whole_match():
    pattern = re.compile(r'"value":\s*"(READY)"')
    text = '{"value": "READY"}'
    assert find_matches(pattern, text) == ['"value": "READY"']


# ── dedupe (plan §6) ──────────────────────────────────────────────────────


def test_filter_new_fires_first_occurrence_and_suppresses_repeat_when_off():
    seen: set[str] = set()
    first = filter_new(["a"], seen, repeat_trigger=False)
    second = filter_new(["a"], seen, repeat_trigger=False)
    assert first == ["a"]
    assert second == []
    assert seen == {"a"}


def test_filter_new_always_fires_when_repeat_trigger_on():
    seen: set[str] = set()
    first = filter_new(["a"], seen, repeat_trigger=True)
    second = filter_new(["a"], seen, repeat_trigger=True)
    assert first == ["a"]
    assert second == ["a"]
    # repeat_trigger=True never touches the dedupe set at all.
    assert seen == set()


def test_filter_new_handles_multiple_matches_in_one_poll_independently():
    seen: set[str] = set()
    # Two distinct matches in the same poll, plus a repeat of the first later.
    result = filter_new(["a", "b"], seen, repeat_trigger=False)
    assert result == ["a", "b"]

    result2 = filter_new(["a", "c"], seen, repeat_trigger=False)
    # "a" suppressed (already seen), "c" is new.
    assert result2 == ["c"]


def test_filter_new_known_limitation_identical_matched_text_from_different_items():
    """Plan §6/§13: two genuinely different items with a byte-identical matched
    substring are indistinguishable to this dedupe — the second is suppressed
    when repeat_trigger is False. Documented as an accepted limitation, not a
    bug; this test pins the documented behavior."""
    seen: set[str] = set()
    filter_new(["duplicate-text"], seen, repeat_trigger=False)
    result = filter_new(["duplicate-text"], seen, repeat_trigger=False)
    assert result == []
