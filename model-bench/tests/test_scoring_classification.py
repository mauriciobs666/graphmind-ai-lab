"""`modelbench.scoring.classification` — guard-judge's `ItemScorer` (S4 spec §5.1.3, §7 Step 1).

Order, red -> green, matches the spec's own step sequence (§7 Step 1):

1. `_normalize_turns`/`_render_judge_user` unit tests against hand-built rows — the raw-message-
   shape input case (`msgId`/`displayName`, no `speaker` key) is the exact defect class S4 spec
   §2.3 finding 2 exists to catch: production normalizes turns (`guards._recent_turns`) before
   rendering (`app._render_judge_user`); skipping the normalization step would read
   `t.get('speaker', 'member')` against a row with no `speaker` key and silently render every turn
   as `"member"`.
2. `extract_own_line_json_object` unit tests — vectors reused from falkor-chat's own
   `server/tests/test_app.py`/`server/tests/eval/test_judge.py`, confirming this port matches the
   source's documented, tested behaviour.
3. `build_messages` unit tests against one real `guard-judge-understanding` item (post-copy).
4. `score_item` unit tests.
5. `aggregate` unit tests over a small synthetic 85-item-shaped fixture (10/8/4 per tier).
6. `packs.validate_pack` on the real, shipped pack passes clean (`validate --strict` itself is a
   pre-existing, unconditional `NotImplementedError` deferral — `cli.py:20-25`, S2's own runner-spec
   §9 gap, untouched by this pack — so this asserts the underlying check directly).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from modelbench.lmstudio import ChatResult
from modelbench.packs import Pack, load_pack, validate_pack
from modelbench.results import ItemTiming
from modelbench.scoring import classification

_REAL_PACK_ROOT = Path(__file__).resolve().parents[1] / "packs" / "guard-judge-understanding"


# ==================================================================================================
# 1. `_normalize_turns` (transcribed from `guards._recent_turns`, guards.py:538-567)
# ==================================================================================================


class TestNormalizeTurns:
    def test_filters_rows_with_no_text_before_slicing_to_the_last_six(self):
        """K-027 carried finding, `guards._recent_turns`'s own docstring rule: malformed/empty rows
        are dropped FIRST, and only then is the tail taken. Eight rows, one of them textless — the
        last six VALID rows must survive, never the last six raw rows (which would drop t3)."""
        rows = [{"text": f"t{i}", "displayName": "Alice", "role": "user"} for i in range(1, 8)]
        rows.insert(1, {"text": "", "displayName": "Bob", "role": "user"})  # empty text, dropped
        turns = classification._normalize_turns(rows)
        assert [t["text"] for t in turns] == ["t2", "t3", "t4", "t5", "t6", "t7"]

    def test_drops_rows_with_missing_text_key(self):
        rows = [{"displayName": "Bob", "role": "user"}, {"text": "kept", "displayName": "Alice"}]
        turns = classification._normalize_turns(rows)
        assert turns == [{"speaker": "Alice", "role": "user", "text": "kept"}]

    def test_drops_rows_with_non_string_text(self):
        rows = [{"text": 42, "displayName": "Bob"}, {"text": "kept", "displayName": "Alice"}]
        turns = classification._normalize_turns(rows)
        assert [t["text"] for t in turns] == ["kept"]

    def test_raw_message_shape_input_normalizes_speaker_from_displayname(self):
        """The exact defect class §2.3 finding 2 exists to catch: `golden_guards.jsonl`'s `turns`
        rows carry the raw `repository.read_thread` shape (`msgId`/`displayName`/`authorId`/
        `createdAt`/`authorType`) — no `speaker` key at all. Skipping this normalization and
        rendering such a row directly would read `t.get('speaker', 'member')` and silently render
        every turn as "member"."""
        row = {
            "msgId": "m-1", "text": "hello", "role": "user", "createdAt": 1,
            "authorId": "u-alice", "displayName": "Alice", "authorType": ["User"],
        }
        turns = classification._normalize_turns([row])
        assert turns == [{"speaker": "Alice", "role": "user", "text": "hello"}]

    def test_speaker_falls_back_to_author_id_when_displayname_absent(self):
        row = {"text": "hi", "authorId": "u-bob"}
        assert classification._normalize_turns([row])[0]["speaker"] == "u-bob"

    def test_speaker_falls_back_to_member_when_neither_present(self):
        row = {"text": "hi"}
        assert classification._normalize_turns([row])[0]["speaker"] == "member"

    def test_role_falls_back_to_user_when_absent(self):
        row = {"text": "hi", "displayName": "Alice"}
        assert classification._normalize_turns([row])[0]["role"] == "user"

    def test_truncates_text_to_400_chars(self):
        row = {"text": "x" * 500, "displayName": "Alice"}
        assert len(classification._normalize_turns([row])[0]["text"]) == 400

    def test_keeps_last_six_when_more_than_six_valid_rows(self):
        rows = [{"text": f"t{i}", "displayName": "Alice"} for i in range(1, 10)]
        turns = classification._normalize_turns(rows)
        assert [t["text"] for t in turns] == ["t4", "t5", "t6", "t7", "t8", "t9"]

    def test_empty_input_yields_empty_list(self):
        assert classification._normalize_turns([]) == []


# ==================================================================================================
# 1 (cont'd). `_render_judge_user` (transcribed from `app._render_judge_user`, app.py:619-654)
# ==================================================================================================


class TestRenderJudgeUser:
    def test_condition_only_when_both_understanding_and_turns_empty(self):
        rendered = classification._render_judge_user("cond text", {}, [])
        assert rendered == "CONDITION: cond text"

    def test_includes_current_state_block_when_understanding_non_empty(self):
        rendered = classification._render_judge_user("cond", {"request": "x"}, [])
        assert "CONDITION: cond" in rendered
        expected_state = json.dumps({"request": "x"}, indent=2, sort_keys=True, default=str)
        assert f"CURRENT STATE:\n{expected_state}" in rendered

    def test_includes_recent_turns_block_when_turns_non_empty(self):
        turns = [{"speaker": "Alice", "role": "user", "text": "hi"}]
        rendered = classification._render_judge_user("cond", {}, turns)
        assert "RECENT TURNS (context only):" in rendered
        assert "Alice: hi" in rendered

    def test_omits_current_state_block_when_understanding_is_empty_dict(self):
        turns = [{"speaker": "a", "role": "user", "text": "hi"}]
        rendered = classification._render_judge_user("cond", {}, turns)
        assert "CURRENT STATE" not in rendered

    def test_omits_recent_turns_block_when_turns_is_empty_list(self):
        rendered = classification._render_judge_user("cond", {"k": "v"}, [])
        assert "RECENT TURNS" not in rendered

    def test_eviction_by_suffix_truncation_drops_oldest_turns_first(self):
        """Mirrors `_render_judge_user`'s own `while kept > 0 and total > JUDGE_USER_MAX_CHARS`
        loop (app.py:643-654) at a synthetic char budget: oldest turns are dropped first, so the
        newest turn always survives."""
        turns = [
            {"speaker": "Alice", "role": "user", "text": "a" * 3000},
            {"speaker": "Bob", "role": "user", "text": "b" * 3000},
            {"speaker": "Carol", "role": "user", "text": "c" * 3000},
        ]
        rendered = classification._render_judge_user("cond", {}, turns)
        assert len(rendered) <= classification._JUDGE_USER_MAX_CHARS
        assert "c" * 3000 in rendered  # newest survives
        assert "a" * 3000 not in rendered  # oldest evicted first
        assert "b" * 3000 not in rendered

    def test_no_turns_fit_falls_back_to_a_hard_base_truncation(self):
        """When even the single newest turn cannot fit, `kept` reaches 0 and the function falls
        back to truncating `base` itself to `JUDGE_USER_MAX_CHARS` (app.py:654)."""
        huge_condition = "x" * 7000
        turns = [{"speaker": "Alice", "role": "user", "text": "y" * 7000}]
        rendered = classification._render_judge_user(huge_condition, {}, turns)
        assert len(rendered) == classification._JUDGE_USER_MAX_CHARS
        assert "RECENT TURNS" not in rendered


# ==================================================================================================
# 2. `extract_own_line_json_object` (transcribed from `llm.py:540-601`) — vectors reused from
# falkor-chat's own `server/tests/test_app.py` (`_judge_verdict`'s own cases) and
# `server/tests/eval/test_judge.py`, confirming this port matches the source's documented, tested
# behaviour rather than a fresh guess at it.
# ==================================================================================================


class TestExtractOwnLineJsonObject:
    def test_bare_object_parses(self):
        parsed = classification.extract_own_line_json_object(
            '{"decision": true, "rationale": "all fields present"}'
        )
        assert parsed == {"decision": True, "rationale": "all fields present"}

    def test_fenced_json_object_parses(self):
        parsed = classification.extract_own_line_json_object(
            '```json\n{"decision": true, "rationale": "the user named the service"}\n```'
        )
        assert parsed == {"decision": True, "rationale": "the user named the service"}

    def test_unlabelled_fenced_object_parses(self):
        parsed = classification.extract_own_line_json_object(
            '```\n{"decision": true, "rationale": "all fields given"}\n```'
        )
        assert parsed["decision"] is True

    def test_prose_wrapped_own_line_object_parses(self):
        parsed = classification.extract_own_line_json_object(
            'Here is my verdict:\n{"decision": true, "rationale": "the user gave repro steps"}\n'
            "Let me know if you need more."
        )
        assert parsed["decision"] is True

    def test_json_that_is_not_an_object_is_none(self):
        assert classification.extract_own_line_json_object("[1, 2, 3]") is None
        assert classification.extract_own_line_json_object('"just a string"') is None

    def test_prose_with_no_object_is_none(self):
        assert (
            classification.extract_own_line_json_object("Yes, the condition is clearly satisfied.")
            is None
        )

    def test_quoted_mid_sentence_verdict_is_rejected(self):
        """gate B-1: an object merely quoted inside a sentence — its `{` is not the first
        non-whitespace character of its line — must not be lifted out as if asserted."""
        reply = (
            "If the user had named the service I would answer "
            '{"decision": true, "rationale": "named"} but they did not, so I answer false.'
        )
        assert classification.extract_own_line_json_object(reply, require_key="decision") is None

    def test_inline_schema_echo_is_rejected(self):
        reply = (
            'The expected reply shape is {"decision": true, "rationale": "..."} — '
            "in this case the condition is not met."
        )
        assert classification.extract_own_line_json_object(reply, require_key="decision") is None

    def test_own_line_schema_echo_is_accepted(self):
        """Declared residual (llm.py's own docstring): an object that owns its lines is accepted
        even when it reads as hypothetical prose around it — line ownership is the only signal a
        parser has."""
        reply = (
            "The reply shape is:\n"
            '{"decision": true, "rationale": "..."}\n'
            "In this case the condition is not met."
        )
        parsed = classification.extract_own_line_json_object(reply, require_key="decision")
        assert parsed["decision"] is True

    def test_two_candidate_objects_disagreeing_resolves_to_none(self):
        reply = (
            "Example of advancing:\n"
            '{"decision": true, "rationale": "all fields given"}\n'
            "My actual answer:\n"
            '{"decision": false, "rationale": "service not named"}'
        )
        assert classification.extract_own_line_json_object(reply, require_key="decision") is None

    def test_two_ambiguous_candidate_objects_with_no_require_key_also_resolves_to_none(self):
        reply = (
            '{"faithfulness": true, "relevance": true}\n'
            '{"faithfulness": false, "relevance": false}'
        )
        assert classification.extract_own_line_json_object(reply) is None

    def test_own_line_object_without_the_required_key_is_ignored(self):
        reply = (
            "Here is the evidence I considered:\n"
            '{"request": "access", "known": ["service"]}\n'
            "I cannot decide."
        )
        assert classification.extract_own_line_json_object(reply, require_key="decision") is None

    def test_require_key_disambiguates_among_multiple_own_line_objects(self):
        reply = (
            '{"faithfulness": true}\n'
            "some narration in between\n"
            '{"faithfulness": true, "relevance": true}'
        )
        parsed = classification.extract_own_line_json_object(reply, require_key="relevance")
        assert parsed == {"faithfulness": True, "relevance": True}

    def test_whole_reply_object_ignores_require_key(self):
        """`require_key` only disambiguates the own-line-scan branch (per its own docstring) — a
        reply that IS ENTIRELY one JSON object parses unconditionally, even missing the key."""
        parsed = classification.extract_own_line_json_object(
            '{"faithfulness": true}', require_key="relevance"
        )
        assert parsed == {"faithfulness": True}

    def test_array_wrapped_verdict_on_one_line_is_none(self):
        """A non-dict top-level value is rejected outright — `extract_own_line_json_object` only
        ever returns a dict or `None`."""
        assert (
            classification.extract_own_line_json_object(
                '[{"decision": true, "rationale": "named"}]', require_key="decision"
            )
            is None
        )

    def test_array_wrapped_verdict_across_multiple_lines_parses_the_inner_own_line_object(self):
        """The outer `[`/`]` never parses as a dict, but once the array spans multiple lines the
        inner object opens ITS OWN line (`^[ \\t]*\\{`, MULTILINE) — own-line scanning finds it
        independent of what brackets surround it (llm.py's own asymmetry with the single-line
        case above; pinned by `test_app.py`'s own
        `test_build_llm_judge_advances_on_a_multiline_array_wrapped_verdict`)."""
        parsed = classification.extract_own_line_json_object(
            '[\n{"decision": true, "rationale": "named"}\n]', require_key="decision"
        )
        assert parsed == {"decision": True, "rationale": "named"}

    def test_non_string_content_is_none(self):
        assert classification.extract_own_line_json_object(None) is None
        assert classification.extract_own_line_json_object(42) is None

    def test_empty_string_is_none(self):
        assert classification.extract_own_line_json_object("") is None
        assert classification.extract_own_line_json_object("   ") is None


# ==================================================================================================
# 3. `build_messages` against one real `guard-judge-understanding` item, post-copy (§7 Step 1)
# ==================================================================================================


def _load_real_items() -> list[dict[str, Any]]:
    with (_REAL_PACK_ROOT / "items.jsonl").open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="module")
def real_pack() -> Pack:
    if not (_REAL_PACK_ROOT / "items.jsonl").exists():
        pytest.skip(
            f"{_REAL_PACK_ROOT / 'items.jsonl'} does not exist — run "
            "`scripts/refresh_golden.py --pack packs/guard-judge-understanding` first"
        )
    return load_pack(_REAL_PACK_ROOT)


class TestBuildMessages:
    def test_system_message_equals_the_prompt_files_content_exactly(self, real_pack: Pack):
        item = next(r for r in _load_real_items() if r["path"] == "understanding")
        messages = classification.build_messages(item, pack=real_pack)
        expected = (real_pack.root / "prompts" / "judge.md").read_text(encoding="utf-8")
        assert messages[0] == {"role": "system", "content": expected}

    def test_user_message_matches_a_hand_computed_render_for_an_understanding_path_item(
        self, real_pack: Pack
    ):
        item = next(r for r in _load_real_items() if r["itemId"] == "ca-01")
        messages = classification.build_messages(item, pack=real_pack)
        expected_user = classification._render_judge_user(
            item["condition"], item["understanding"], []
        )
        assert messages[1] == {"role": "user", "content": expected_user}

    def test_user_message_normalizes_turns_for_a_turns_path_item(self, real_pack: Pack):
        """The exact defect class §2.3 finding 2 exists to catch, now at `build_messages`'s own
        seam: `tn-01`'s raw `turns` rows carry `msgId`/`displayName`, no `speaker` key — the
        rendered user message must show the real display names, never a blanket "member" for
        every turn."""
        item = next(r for r in _load_real_items() if r["itemId"] == "tn-01")
        messages = classification.build_messages(item, pack=real_pack)
        expected_turns = classification._normalize_turns(item["turns"])
        expected_user = classification._render_judge_user(
            item["condition"], item["understanding"], expected_turns
        )
        assert messages[1] == {"role": "user", "content": expected_user}
        assert "Alice: " in messages[1]["content"]
        assert "member:" not in messages[1]["content"]

    def test_never_reads_the_gold_label_fields(self, real_pack: Pack):
        """`build_messages` must never read `expected`/`label_rationale`/`r1_probe` — those would
        leak the gold label into the prompt. Constructed by deleting the three keys before
        calling: must not raise `KeyError`."""
        item = dict(next(r for r in _load_real_items() if r["path"] == "understanding"))
        del item["expected"]
        del item["label_rationale"]
        del item["r1_probe"]
        messages = classification.build_messages(item, pack=real_pack)  # must not raise
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


# ==================================================================================================
# 4. `score_item` (transcribed fallback from `app.py:679-682`/`726-729`)
# ==================================================================================================


def _chat_result(content: str) -> ChatResult:
    return ChatResult(
        message={"role": "assistant", "content": content},
        tool_calls=(),
        toolCallForm="prose",
        stats=None,
        model_info=None,
        runtime=None,
        usage=None,
        wallClockMs=5.0,
    )


def _timing() -> ItemTiming:
    return ItemTiming(wallClockMs=5.0, calls=(), withheldFor=None)


def _timeout_timing() -> ItemTiming:
    return ItemTiming(wallClockMs=None, calls=(), withheldFor="timeout")


def _no_response_timing() -> ItemTiming:
    return ItemTiming(wallClockMs=None, calls=(), withheldFor="no_response")


class TestScoreItem:
    def test_timeout_scores_fail_with_no_scoreable_metrics(self):
        item_input = {"itemId": "x", "tier": "clear_suspend", "path": "understanding"}
        result = classification.score_item(item_input, None, _timeout_timing(), pack=None)
        assert result.outcome == "fail"
        assert result.scoreable == {}
        assert result.counts == {}

    def test_no_response_scores_unrunnable_with_no_scoreable_metrics(self):
        item_input = {"itemId": "x", "tier": "clear_suspend", "path": "understanding"}
        result = classification.score_item(item_input, None, _no_response_timing(), pack=None)
        assert result.outcome == "unrunnable"
        assert result.scoreable == {}
        assert result.counts == {}

    def test_parse_failure_falls_back_to_the_same_bias_to_suspend_default_the_real_judge_applies(
        self,
    ):
        """`app.py:727-728`'s own fallback: an unparseable reply resolves to `decision=False`,
        never a fabricated guess — reproduced here so the item still contributes to its metric's
        denominator with the SAME verdict the real judge would have recorded."""
        item_input = {"itemId": "x", "tier": "clear_suspend", "path": "understanding"}
        result = classification.score_item(
            item_input, _chat_result("not json at all"), _timing(), pack=None
        )
        assert result.outcome == "parse_failure"
        # clear_suspend's own metric is falseAdvanceRate; decision=False -> advanced=False ->
        # NOT a false advance -> count 0 (mirrors the real judge's bias-to-suspend fallback).
        assert result.counts["falseAdvanceRate"] == 0
        assert result.scoreable == {"falseAdvanceRate": True}

    def test_clear_suspend_tier_advancing_counts_a_false_advance(self):
        """`clear_suspend`'s items are all `expected=False`; the judge advancing anyway (`decision:
        true`) IS the false-advance event `falseAdvanceRate` counts."""
        item_input = {"itemId": "cs-01", "tier": "clear_suspend", "path": "understanding"}
        reply = _chat_result('{"decision": true, "rationale": "looks satisfied"}')
        result = classification.score_item(item_input, reply, _timing(), pack=None)
        assert result.outcome == "pass"
        assert result.counts == {"falseAdvanceRate": 1}
        assert result.scoreable == {"falseAdvanceRate": True}

    def test_clear_suspend_tier_correctly_suspending_counts_no_false_advance(self):
        item_input = {"itemId": "cs-02", "tier": "clear_suspend", "path": "understanding"}
        reply = _chat_result('{"decision": false, "rationale": "missing info"}')
        result = classification.score_item(item_input, reply, _timing(), pack=None)
        assert result.counts == {"falseAdvanceRate": 0}

    def test_clear_advance_tier_suspending_counts_a_false_suspend(self):
        """`clear_advance`'s items are all `expected=True`; the judge suspending anyway (`decision:
        false`) IS the false-suspend event `falseSuspendRate` counts."""
        item_input = {"itemId": "ca-01", "tier": "clear_advance", "path": "understanding"}
        reply = _chat_result('{"decision": false, "rationale": "not enough"}')
        result = classification.score_item(item_input, reply, _timing(), pack=None)
        assert result.outcome == "pass"
        assert result.counts == {"falseSuspendRate": 1}
        assert result.scoreable == {"falseSuspendRate": True}

    def test_clear_advance_tier_correctly_advancing_counts_no_false_suspend(self):
        item_input = {"itemId": "ca-02", "tier": "clear_advance", "path": "understanding"}
        reply = _chat_result('{"decision": true, "rationale": "looks complete"}')
        result = classification.score_item(item_input, reply, _timing(), pack=None)
        assert result.counts == {"falseSuspendRate": 0}

    def test_boundary_tier_advancing_counts_a_false_advance_boundary(self):
        """`boundary`'s items are all `expected=False` (§2.4, verified against the real 85-row
        set); advancing is exploratory-only (`falseAdvanceRateBoundary`, never a verdict metric)."""
        item_input = {"itemId": "bd-01", "tier": "boundary", "path": "understanding"}
        reply = _chat_result('{"decision": true, "rationale": "edge case"}')
        result = classification.score_item(item_input, reply, _timing(), pack=None)
        assert result.outcome == "pass"
        assert result.counts == {"falseAdvanceRateBoundary": 1}
        assert result.scoreable == {"falseAdvanceRateBoundary": True}

    def test_detail_carries_path_and_tier_never_anything_answer_adjacent(self):
        item_input = {
            "itemId": "cs-03", "tier": "clear_suspend", "path": "turns",
            "expected": False, "label_rationale": "secret",
        }
        reply = _chat_result('{"decision": false, "rationale": "fine"}')
        result = classification.score_item(item_input, reply, _timing(), pack=None)
        assert result.detail == {"path": "turns", "tier": "clear_suspend"}


# ==================================================================================================
# 5. `aggregate` over a small synthetic 22-item (10/8/4 per tier) fixture, scaled down from 85
# ==================================================================================================


def _scored(item_id: str, tier: str, path: str, decision: bool | None) -> Any:
    """`decision=None` builds a parse-failure reply (unparseable content); otherwise a clean
    `{"decision": <bool>, ...}` reply. Routes through the real `score_item`, never a hand-built
    `ItemResult`, so the fixture exercises the same path this module's own runtime does."""
    item_input = {"itemId": item_id, "tier": tier, "path": path}
    reply = (
        _chat_result("not json at all")
        if decision is None
        else _chat_result(json.dumps({"decision": decision, "rationale": "r"}))
    )
    return classification.score_item(item_input, reply, _timing(), pack=None)


def _synthetic_items() -> list[Any]:
    # clear_suspend (metric=falseAdvanceRate, n=10): 6 understanding + 4 turns.
    # Advances (successes): cs-u1, cs-u2, cs-t1 = 3 total (2 understanding, 1 turns).
    # cs-u6 is an unparseable reply (parse failure -> advanced=False, still counted, n unaffected).
    clear_suspend = [
        _scored("cs-u1", "clear_suspend", "understanding", True),
        _scored("cs-u2", "clear_suspend", "understanding", True),
        _scored("cs-u3", "clear_suspend", "understanding", False),
        _scored("cs-u4", "clear_suspend", "understanding", False),
        _scored("cs-u5", "clear_suspend", "understanding", False),
        _scored("cs-u6", "clear_suspend", "understanding", None),
        _scored("cs-t1", "clear_suspend", "turns", True),
        _scored("cs-t2", "clear_suspend", "turns", False),
        _scored("cs-t3", "clear_suspend", "turns", False),
        _scored("cs-t4", "clear_suspend", "turns", False),
    ]
    # clear_advance (metric=falseSuspendRate, n=8): 5 understanding + 3 turns.
    # False suspends (decision=False, successes): ca-u3, ca-t1 = 2 total (1 understanding, 1 turns).
    clear_advance = [
        _scored("ca-u1", "clear_advance", "understanding", True),
        _scored("ca-u2", "clear_advance", "understanding", True),
        _scored("ca-u3", "clear_advance", "understanding", False),
        _scored("ca-u4", "clear_advance", "understanding", True),
        _scored("ca-u5", "clear_advance", "understanding", True),
        _scored("ca-t1", "clear_advance", "turns", False),
        _scored("ca-t2", "clear_advance", "turns", True),
        _scored("ca-t3", "clear_advance", "turns", True),
    ]
    # boundary (metric=falseAdvanceRateBoundary, n=4): successes bd1, bd3 = 2.
    boundary = [
        _scored("bd1", "boundary", "understanding", True),
        _scored("bd2", "boundary", "understanding", False),
        _scored("bd3", "boundary", "turns", True),
        _scored("bd4", "boundary", "turns", False),
    ]
    return clear_suspend + clear_advance + boundary


class TestAggregate:
    def test_per_class_carries_the_two_verdict_metrics_at_their_own_tiers_n_never_85(self):
        aggregates = classification.aggregate(_synthetic_items(), pack=None)
        by_name = {m.name: m for m in aggregates.perClass}

        false_advance = by_name["falseAdvanceRate"]
        assert false_advance.n == 10
        assert false_advance.successes == 3

        false_suspend = by_name["falseSuspendRate"]
        assert false_suspend.n == 8
        assert false_suspend.successes == 2

        # Neither is ever the pooled 22-item (or 85-item) total.
        assert false_advance.n != len(_synthetic_items())
        assert false_suspend.n != len(_synthetic_items())

    def test_advance_recall_is_the_exact_complement_of_false_suspend_rate(self):
        aggregates = classification.aggregate(_synthetic_items(), pack=None)
        by_name = {m.name: m for m in aggregates.perClass}
        false_suspend = by_name["falseSuspendRate"]
        advance_recall = by_name["advanceRecall"]
        assert advance_recall.n == false_suspend.n
        assert advance_recall.successes == false_suspend.n - false_suspend.successes

    def test_boundary_tier_metric_is_exploratory_at_its_own_n(self):
        aggregates = classification.aggregate(_synthetic_items(), pack=None)
        by_name = {m.name: m for m in aggregates.perClass}
        boundary = by_name["falseAdvanceRateBoundary"]
        assert boundary.n == 4
        assert boundary.successes == 2

    def test_path_split_metrics_sum_back_to_their_parent_tiers_own_successes_and_n(self):
        aggregates = classification.aggregate(_synthetic_items(), pack=None)
        by_name = {m.name: m for m in aggregates.perClass}

        false_advance = by_name["falseAdvanceRate"]
        by_understanding = by_name["falseAdvanceRateByUnderstanding"]
        by_turns = by_name["falseAdvanceRateByTurns"]
        assert by_understanding.n == 6
        assert by_understanding.successes == 2
        assert by_turns.n == 4
        assert by_turns.successes == 1
        assert by_understanding.n + by_turns.n == false_advance.n
        assert by_understanding.successes + by_turns.successes == false_advance.successes

        false_suspend = by_name["falseSuspendRate"]
        fs_by_understanding = by_name["falseSuspendRateByUnderstanding"]
        fs_by_turns = by_name["falseSuspendRateByTurns"]
        assert fs_by_understanding.n == 5
        assert fs_by_understanding.successes == 1
        assert fs_by_turns.n == 3
        assert fs_by_turns.successes == 1
        assert fs_by_understanding.n + fs_by_turns.n == false_suspend.n
        assert fs_by_understanding.successes + fs_by_turns.successes == false_suspend.successes

    def test_parse_failures_counts_only_parse_failure_outcomes(self):
        aggregates = classification.aggregate(_synthetic_items(), pack=None)
        assert aggregates.parseFailures == 1  # only cs-u6

    def test_n_is_the_total_item_count(self):
        items = _synthetic_items()
        aggregates = classification.aggregate(items, pack=None)
        assert aggregates.n == len(items) == 22

    def test_every_binary_metric_uses_the_item_unit(self):
        aggregates = classification.aggregate(_synthetic_items(), pack=None)
        assert all(m.unit == "item" for m in aggregates.perClass)


# ==================================================================================================
# Real-pack sanity check — the shipped 85-item golden set's own tier sizes (S4 spec §5.1.3's own
# cited n values: falseAdvanceRate n=40, falseSuspendRate n=30, falseAdvanceRateBoundary n=15)
# ==================================================================================================


def test_aggregate_over_the_real_85_item_pack_matches_the_specs_own_tier_ns(real_pack: Pack):
    items = _load_real_items()
    scored = []
    for item in items:
        reply = _chat_result('{"decision": false, "rationale": "r"}')
        scored.append(classification.score_item(item, reply, _timing(), pack=real_pack))
    aggregates = classification.aggregate(scored, pack=real_pack)
    by_name = {m.name: m for m in aggregates.perClass}
    assert aggregates.n == 85
    assert by_name["falseAdvanceRate"].n == 40
    assert by_name["falseSuspendRate"].n == 30
    assert by_name["falseAdvanceRateBoundary"].n == 15
    assert by_name["advanceRecall"].n == 30


# ==================================================================================================
# 6. `packs.validate_pack` on the real, shipped pack (§7 Step 1's own done-condition). `validate
# --strict` itself is a pre-existing, unconditional `NotImplementedError` deferral (`cli.py:20-25`,
# S2's own runner-spec §9 gap) — untouched by this pack and orthogonal to its own validity, so this
# asserts the underlying check `validate --strict` would still have to run underneath, once built.
# ==================================================================================================


def test_the_real_shipped_pack_validates_clean():
    pack = load_pack(_REAL_PACK_ROOT)
    assert validate_pack(pack) == []


# ==================================================================================================
# 7. typed-decision-models U3a — `score_item` split into `answer_from_chat` + `score_answer`
#    (docs/plans/typed-decision-models.md §3.3 "Scorer seam", §4 U3a), with no behaviour change
# ==================================================================================================

#: The reply corpus the split is re-scored over. No stored record keeps a model's reply text
#: (`model-bench/AGENTS.md`), so the "fixture replies" are written here: one per branch the parse
#: and the count can take, including the loose ones the split must carry over unchanged — a
#: `"decision": "false"` string is an advance today (`bool("false")`), and a whole-reply object
#: with no `decision` key is a `pass` that suspends. Order and ids are part of the golden.
SPLIT_REPLY_CORPUS: tuple[tuple[str, Any], ...] = (
    ("true", '{"decision": true, "rationale": "looks satisfied"}'),
    ("false", '{"decision": false, "rationale": "missing info"}'),
    ("garbage", "not json at all"),
    ("fenced-true", '```json\n{"decision": true}\n```'),
    ("prose-own-line-true", 'Thinking it over.\n{"decision": true, "rationale": "x"}\n'),
    ("mid-sentence-true", 'I would say {"decision": true} here.'),
    ("string-true", '{"decision": "true"}'),
    ("string-false", '{"decision": "false"}'),
    ("zero", '{"decision": 0}'),
    ("null", '{"decision": null}'),
    ("no-key", '{"rationale": "no decision key"}'),
    ("two-own-line", 'a\n{"decision": true}\n{"decision": false}\n'),
    ("keyless-beside-key", 'Schema:\n{"rationale": "x"}\nAnswer:\n{"decision": true}\n'),
    ("empty", ""),
    ("non-string", None),
)

#: The two `result is None` withholdings, re-scored too: the split must not move them.
SPLIT_WITHHELD = ("timeout", "no_response")

#: Captured from the **unmodified** `score_item` (HEAD `dc2f8884`) before the split, by
#: `split_rescore_lines()` below; see `docs/HISTORY.md` (U3a).
_SPLIT_GOLDEN = (
    Path(__file__).parent / "fixtures" / "classification" / "score_item_before_split.json"
)


def _split_timing(withheld: str | None) -> ItemTiming:
    from modelbench.results import CallTiming

    if withheld is not None:
        return ItemTiming(wallClockMs=None, calls=(), withheldFor=withheld)
    call = CallTiming(
        wallClockMs=612.5, ttftMs=101.0, generationMs=480.25, promptTokens=812,
        tokensPerSecond=41.5,
    )
    return ItemTiming(wallClockMs=640.0, calls=(call,), withheldFor=None)


def _split_chat_result(content: Any) -> ChatResult:
    return ChatResult(
        message={"role": "assistant", "content": content},
        tool_calls=(),
        toolCallForm="prose",
        stats=None,
        model_info=None,
        runtime=None,
        usage=None,
        wallClockMs=612.5,
    )


def split_rescore_lines(pack: Pack) -> dict[str, list[str]]:
    """Every real pack item × every corpus reply (and both withholdings), scored through
    `classification.score_item`, as `{variant: [repr(ItemResult), ...]}` in item order. `repr`,
    not `to_dict`: it keeps a `True` count distinct from a `1`, which JSON would not."""
    items = _load_real_items()
    out: dict[str, list[str]] = {}
    for reply_id, content in SPLIT_REPLY_CORPUS:
        out[reply_id] = [
            repr(
                classification.score_item(
                    item, _split_chat_result(content), _split_timing(None), pack=pack
                )
            )
            for item in items
        ]
    for withheld in SPLIT_WITHHELD:
        out[f"withheld-{withheld}"] = [
            repr(classification.score_item(item, None, _split_timing(withheld), pack=pack))
            for item in items
        ]
    return out


def split_digests(lines: dict[str, list[str]]) -> dict[str, Any]:
    import hashlib

    def digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    return {
        "cases": sum(len(v) for v in lines.values()),
        "perVariant": {k: digest("\n".join(v)) for k, v in lines.items()},
        "all": digest("\n".join(f"{k}\t{line}" for k, v in lines.items() for line in v)),
    }


def test_score_item_rescores_the_real_pack_byte_identically_to_before_the_split(real_pack: Pack):
    """Plan §4 U3a's done-condition: the old pack, re-scored after the split, gives `ItemResult`s
    byte-identical to the unmodified scorer's. 85 items × 17 variants = 1445 cases."""
    golden = json.loads(_SPLIT_GOLDEN.read_text(encoding="utf-8"))
    actual = split_digests(split_rescore_lines(real_pack))
    assert actual["cases"] == golden["cases"] == 85 * (len(SPLIT_REPLY_CORPUS) + 2)
    assert actual["perVariant"] == golden["perVariant"]
    assert actual["all"] == golden["all"]


class TestAnswerFromChat:
    """`answer_from_chat(result, *, pack) -> (advanced, outcome)` — the role's chat parse, alone."""

    def test_an_asserted_true_decision_is_an_advance(self):
        result = _chat_result('{"decision": true, "rationale": "ok"}')
        assert classification.answer_from_chat(result, pack=None) == (True, "pass")

    def test_an_asserted_false_decision_is_a_suspend(self):
        result = _chat_result('{"decision": false, "rationale": "no"}')
        assert classification.answer_from_chat(result, pack=None) == (False, "pass")

    def test_an_unparseable_reply_suspends_as_a_parse_failure(self):
        result = _chat_result("not json at all")
        assert classification.answer_from_chat(result, pack=None) == (False, "parse_failure")

    def test_the_old_packs_loose_parse_is_kept_a_false_string_is_an_advance(self):
        """Plan §2.1: this pack's parse is `bool(parsed.get("decision"))`, looser than
        falkor-chat's `_coerce_verdict`. U3a changes nothing; the strict parse is U3b's, for
        decision-point packs only."""
        result = _chat_result('{"decision": "false"}')
        assert classification.answer_from_chat(result, pack=None) == (True, "pass")


#: The old pack's call: every decision-surface keyword `None` (plan2 §3.3, plan3 §3.3).
_NO_DECISION = {
    "decided_by": None, "probability": None, "failed_call": None, "reason": None,
    "input_tokens": None,
}


class TestScoreAnswer:
    """`score_answer(item_input, answer, outcome, timing, *, pack, decided_by, probability,
    failed_call, reason) -> ItemResult` — the tier's metric and count for an answer, wherever the
    answer came from."""

    def test_an_advance_on_a_clear_suspend_item_counts_a_false_advance(self):
        item_input = {"itemId": "cs-01", "tier": "clear_suspend", "path": "turns"}
        result = classification.score_answer(
            item_input, True, "pass", _timing(), pack=None, **_NO_DECISION
        )
        assert result.outcome == "pass"
        assert result.scoreable == {"falseAdvanceRate": True}
        assert result.counts == {"falseAdvanceRate": 1}
        assert result.detail == {"path": "turns", "tier": "clear_suspend"}
        assert result.timing == _timing()

    def test_a_suspend_on_a_clear_advance_item_counts_a_false_suspend(self):
        item_input = {"itemId": "ca-01", "tier": "clear_advance", "path": "understanding"}
        result = classification.score_answer(
            item_input, False, "pass", _timing(), pack=None, **_NO_DECISION
        )
        assert result.counts == {"falseSuspendRate": 1}

    def test_the_outcome_is_carried_through_not_recomputed(self):
        item_input = {"itemId": "bd-01", "tier": "boundary", "path": "turns"}
        result = classification.score_answer(
            item_input, False, "parse_failure", _timing(), pack=None, **_NO_DECISION
        )
        assert result.outcome == "parse_failure"
        assert result.counts == {"falseAdvanceRateBoundary": 0}


# ==================================================================================================
# 8. typed-decision-models U1d — the `score_answer` delta (plan2 §3.3, plan3 §3.3)
# ==================================================================================================

_CS_ITEM = {"itemId": "cs-01", "tier": "clear_suspend", "path": "turns"}


def _decision_keywords(**overrides: Any) -> dict[str, Any]:
    return {**_NO_DECISION, **overrides}


class TestScoreAnswerDecisionSurfaceDelta:
    @pytest.mark.parametrize(
        "name", ["decided_by", "probability", "failed_call", "reason", "input_tokens"]
    )
    def test_each_new_keyword_is_keyword_only_with_no_default(self, name):
        """Plan2 §3.3 / plan3 §3.3: no default — a call site that forgets one is a `TypeError`,
        never an item silently scored as if the decision model had not been there."""
        import inspect

        param = inspect.signature(classification.score_answer).parameters[name]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is inspect.Parameter.empty

    def test_all_four_none_writes_exactly_todays_detail(self):
        result = classification.score_answer(
            _CS_ITEM, True, "pass", _timing(), pack=None, **_NO_DECISION
        )
        assert result.detail == {"path": "turns", "tier": "clear_suspend"}
        assert list(result.detail) == ["path", "tier"]

    def test_the_old_packs_score_item_writes_no_decision_surface_key(self, real_pack: Pack):
        """Plan2 §5 U1d's dedicated test: across every real item and every reply branch, no
        `decidedBy`, `probability`, `failedCall` or `reason` key appears in `detail`."""
        decision_keys = {"decidedBy", "probability", "failedCall", "reason", "inputTokens"}
        for item in _load_real_items():
            for _, content in SPLIT_REPLY_CORPUS:
                result = classification.score_item(
                    item, _split_chat_result(content), _split_timing(None), pack=real_pack
                )
                assert set(result.detail) == {"path", "tier"}
                assert not decision_keys & set(result.detail)

    @pytest.mark.parametrize(
        ("decided_by", "probability"), [("decision-model", 0.931), ("handoff", 0.42)]
    )
    def test_a_decided_item_carries_decided_by_and_probability_and_scores_as_today(
        self, decided_by, probability
    ):
        result = classification.score_answer(
            _CS_ITEM, True, "pass", _timing(), pack=None,
            **_decision_keywords(decided_by=decided_by, probability=probability),
        )
        assert result.detail == {
            "path": "turns", "tier": "clear_suspend",
            "decidedBy": decided_by, "probability": probability, "inputTokens": None,
        }
        assert result.scoreable == {"falseAdvanceRate": True}
        assert result.counts == {"falseAdvanceRate": 1}

    def test_an_over_limit_hand_off_carries_reason_and_a_none_probability(self):
        """Plan3 §3.3's table, "Hand-off succeeds": `decidedBy="handoff"`, `probability=None`,
        `reason="over-limit"`; the hand-off LLM's answer is scored."""
        result = classification.score_answer(
            _CS_ITEM, False, "pass", _timing(), pack=None,
            **_decision_keywords(decided_by="handoff", reason="over-limit"),
        )
        assert result.detail == {
            "path": "turns", "tier": "clear_suspend",
            "decidedBy": "handoff", "probability": None, "inputTokens": None,
            "reason": "over-limit",
        }
        assert result.counts == {"falseAdvanceRate": 0}

    def test_a_failed_decision_call_is_unscoreable_with_failed_call_and_no_decided_by(self):
        """Plan2 §3.3: `failed_call="decision"` gives `scoreable={}` and `detail.failedCall` on
        any pack; the item enters the bar only through `decision_stats.pair_items`."""
        timing = ItemTiming(wallClockMs=2003.0, calls=(), withheldFor=None)
        result = classification.score_answer(
            _CS_ITEM, None, "fail", timing, pack=None,
            **_decision_keywords(failed_call="decision"),
        )
        assert result.scoreable == {}
        assert result.counts == {}
        assert result.outcome == "fail"
        assert result.timing == timing
        assert result.detail == {
            "path": "turns", "tier": "clear_suspend", "failedCall": "decision",
            "probability": None, "inputTokens": None,
        }

    def test_an_over_limit_decision_failure_also_carries_reason(self):
        """Plan3 §3.3, `decisions` surface: the kept decision-call failure plus
        `detail.reason = "over-limit"`."""
        timing = ItemTiming(wallClockMs=40.0, calls=(), withheldFor=None)
        result = classification.score_answer(
            _CS_ITEM, None, "fail", timing, pack=None,
            **_decision_keywords(failed_call="decision", reason="over-limit"),
        )
        assert result.scoreable == {}
        assert result.detail["failedCall"] == "decision"
        assert result.detail["reason"] == "over-limit"
        assert "decidedBy" not in result.detail

    @pytest.mark.parametrize(
        ("answer", "failed_call"), [(None, None), (True, "decision"), (False, "decision")]
    )
    def test_answer_is_none_iff_a_call_failed(self, answer, failed_call):
        with pytest.raises(ValueError, match="answer"):
            classification.score_answer(
                _CS_ITEM, answer, "pass", _timing(), pack=None,
                **_decision_keywords(failed_call=failed_call),
            )

    def test_a_failed_item_has_no_decided_by(self):
        with pytest.raises(ValueError, match="decided_by"):
            classification.score_answer(
                _CS_ITEM, None, "fail", _timing(), pack=None,
                **_decision_keywords(failed_call="decision", decided_by="handoff"),
            )

    def test_a_probability_needs_a_decision_source(self):
        """A probability with no `decided_by` would be dropped from `detail` (the old pack's
        shape) or misattributed to a failed call; it is refused instead."""
        with pytest.raises(ValueError, match="probability"):
            classification.score_answer(
                _CS_ITEM, True, "pass", _timing(), pack=None,
                **_decision_keywords(probability=0.5),
            )

    def test_a_guard_hand_off_failure_is_never_scored_here(self):
        """Plan2 §3.3: `HANDOFF_FAILURE_BY_ROLE["guard-judge"] == "withhold"` — a guard hand-off
        chat-call failure is withheld by the runner, never kept and scored."""
        with pytest.raises(ValueError, match="handoff"):
            classification.score_answer(
                _CS_ITEM, None, "fail", _timing(), pack=None,
                **_decision_keywords(failed_call="handoff"),
            )

    @pytest.mark.parametrize(
        "overrides",
        [{"decided_by": "llm-handoff"}, {"failed_call": "chat"}, {"reason": "too-long"}],
        ids=["decided_by", "failed_call", "reason"],
    )
    def test_each_keyword_is_a_closed_set(self, overrides):
        answer = None if "failed_call" in overrides else True
        with pytest.raises(ValueError, match=next(iter(overrides))):
            classification.score_answer(
                _CS_ITEM, answer, "pass", _timing(), pack=None, **_decision_keywords(**overrides)
            )

    def test_a_decided_item_carries_the_decision_calls_input_tokens(self):
        """Plan2 §3.4: `detail.inputTokens` is the decision call's `usage["input_tokens"]`."""
        result = classification.score_answer(
            _CS_ITEM, True, "pass", _timing(), pack=None,
            **_decision_keywords(decided_by="decision-model", probability=0.9, input_tokens=113),
        )
        assert result.detail["inputTokens"] == 113


# --- P4-m1: every keyword combination, against the plans' legality (impl review Pass 4) ----------


def _legal(answer, decided_by, probability, failed_call, reason, input_tokens, withheld) -> bool:
    """Written from the plans and the architect's ruling, not from `_decision_detail`:
    - plan2 §3.3: `answer` is `None` iff a call failed; a failed item has no `decidedBy`; the
      guard's hand-off chat-call failure is withheld (`HANDOFF_FAILURE_BY_ROLE`), never scored;
    - architect ruling (2026-10-31, U2 seam): the one exception — no answer, no failed call, a
      `reason`, and a withheld timing is the guard's withheld over-limit item (plan3 §3.3's
      `withhold` column), with no `decidedBy`, `failedCall` or `probability`;
    - plan3 §3.3's over-limit table: an over-limit item is handed off (`decidedBy="handoff"`) or a
      kept decision failure, and every such row has `probability = None` and `inputTokens = None`;
    - v1.3 §3.3: a decided item's `probability` is the predicate's p (or a choice's confidence),
      which `decide` always returns — so only an over-limit hand-off is decided without one;
    - plan2 §3.4: `inputTokens` comes from a decision call that returned;
    - the old pack (all keywords `None`) scores an answer, as before."""
    if failed_call == "handoff":
        return False
    if answer is None and failed_call is None:
        return (
            reason == "over-limit" and withheld is not None
            and decided_by is None and probability is None and input_tokens is None
        )
    if (answer is None) != (failed_call is not None):
        return False
    if failed_call == "decision":
        return decided_by is None and probability is None and input_tokens is None
    if decided_by is None:  # an answer from no decision surface: the old pack's call
        return probability is None and reason is None and input_tokens is None
    if reason == "over-limit":
        return decided_by == "handoff" and probability is None and input_tokens is None
    return probability is not None


_COMBINATIONS = [
    (answer, decided_by, probability, failed_call, reason, input_tokens, withheld)
    for answer in (True, None)
    for decided_by in (None, "decision-model", "handoff")
    for probability in (None, 0.5)
    for failed_call in (None, "decision", "handoff")
    for reason in (None, "over-limit")
    for input_tokens in (None, 113)
    for withheld in (None, "timeout")
]


def _combination_timing(withheld: str | None) -> ItemTiming:
    if withheld is None:
        return _timing()
    return ItemTiming(wallClockMs=None, calls=(), withheldFor=withheld)


def test_score_answer_accepts_exactly_the_legal_keyword_combinations():
    """Impl review Pass 4, P4-m1's stopping condition: the reviewer's 72 combinations of
    `answer` × `decided_by` × `probability` × `failed_call` × `reason`, times `input_tokens` ∈
    {None, int} and `timing.withheldFor` ∈ {None, "timeout"} (288), each accepted iff `_legal`
    says so — a refusal is a `ValueError`."""
    assert len(_COMBINATIONS) == 72 * 2 * 2
    wrong = []
    for combo in _COMBINATIONS:
        answer, decided_by, probability, failed_call, reason, input_tokens, withheld = combo
        try:
            classification.score_answer(
                _CS_ITEM, answer, "pass", _combination_timing(withheld), pack=None,
                decided_by=decided_by, probability=probability, failed_call=failed_call,
                reason=reason, input_tokens=input_tokens,
            )
            accepted = True
        except ValueError:
            accepted = False
        if accepted != _legal(*combo):
            wrong.append((*combo, "accepted" if accepted else "refused"))
    assert wrong == []


@pytest.mark.parametrize("withheld", ["timeout", "no_response", "load"])
def test_a_withheld_over_limit_guard_item_carries_only_its_reason(withheld):
    """Architect ruling (plan3 §3.3's guard `withhold` column): no answer, no failed call, a
    reason and a withheld timing give the withheld outcome as passed, nothing scoreable, and
    `detail = {path, tier, reason}` — no `decidedBy`, `failedCall`, `probability`,
    `inputTokens`."""
    timing = ItemTiming(wallClockMs=None, calls=(), withheldFor=withheld)
    outcome = "fail" if withheld == "timeout" else "unrunnable"
    result = classification.score_answer(
        _CS_ITEM, None, outcome, timing, pack=None,
        **_decision_keywords(reason="over-limit"),
    )
    assert result.outcome == outcome
    assert result.scoreable == {}
    assert result.counts == {}
    assert result.timing == timing
    assert result.detail == {"path": "turns", "tier": "clear_suspend", "reason": "over-limit"}


@pytest.mark.parametrize(
    ("reason", "withheld"), [(None, "timeout"), ("over-limit", None)],
    ids=["withheld-without-reason", "reason-without-withheld"],
)
def test_no_answer_and_no_failed_call_needs_both_a_reason_and_a_withheld_timing(reason, withheld):
    with pytest.raises(ValueError, match="answer"):
        classification.score_answer(
            _CS_ITEM, None, "fail", _combination_timing(withheld), pack=None,
            **_decision_keywords(reason=reason),
        )
