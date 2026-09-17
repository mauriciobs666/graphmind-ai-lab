"""`modelbench.scoring.grounding` — `chat-responder`'s `ItemScorer` (S7 spec §3.4, §5 Step 0).

Order, red -> green, mirrors the spec's own §5 Step 0 text: `looks_like_abstention` ->
`resolve_format` -> `checklist_pass` -> `format_checks` (mutation-pair discipline per axis) ->
`build_messages` (its own never-leaks-the-answer-key assertion) -> `score_item` (the `result is
None` branch) -> `aggregate` (four independent `BinaryMetric`s).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from modelbench.lmstudio import ChatResult
from modelbench.packs import Pack, load_pack
from modelbench.results import ItemResult, ItemTiming
from modelbench.scoring import grounding

# ==================================================================================================
# 1. `looks_like_abstention` (transcribed from `nlq_scoring.py`'s `layer2_contains`'s `not_found`
# branch, S7 spec §2.8)
# ==================================================================================================


class TestLooksLikeAbstention:
    def test_detects_an_abstention_marker_case_insensitively(self):
        assert grounding.looks_like_abstention("I DON'T KNOW the answer to that.") is True

    def test_detects_a_marker_across_whitespace_variation(self):
        assert grounding.looks_like_abstention("I   could   not   find  that in the text.") is True

    def test_no_marker_present_returns_false(self):
        assert grounding.looks_like_abstention("The price is 24.99.") is False

    def test_empty_reply_returns_false(self):
        assert grounding.looks_like_abstention("") is False


# ==================================================================================================
# 2. `resolve_format` — the three-way merge (S7 spec §3.1/§3.2/§3.4)
# ==================================================================================================


class TestResolveFormat:
    def test_pack_default_only_when_item_declares_no_format(self):
        pack_format = {
            "maxWords": 150, "mustBeSingleParagraph": True, "forbiddenPatterns": ("```",),
        }
        resolved = grounding.resolve_format(pack_format, None)
        assert resolved == pack_format

    def test_item_overrides_exactly_one_key_others_fall_back_to_pack_default(self):
        pack_format = {
            "maxWords": 150, "mustBeSingleParagraph": True, "forbiddenPatterns": ("```",),
        }
        resolved = grounding.resolve_format(pack_format, {"maxWords": 50})
        assert resolved == {
            "maxWords": 50, "mustBeSingleParagraph": True, "forbiddenPatterns": ("```",),
        }

    def test_item_overrides_all_three_keys(self):
        pack_format = {
            "maxWords": 150, "mustBeSingleParagraph": True, "forbiddenPatterns": ("```",),
        }
        item_format = {
            "maxWords": 30, "mustBeSingleParagraph": False, "forbiddenPatterns": (),
        }
        resolved = grounding.resolve_format(pack_format, item_format)
        assert resolved == item_format

    def test_empty_pack_format_falls_back_to_the_trivially_passing_default(self):
        resolved = grounding.resolve_format({}, None)
        assert resolved == {
            "maxWords": None, "mustBeSingleParagraph": False, "forbiddenPatterns": (),
        }


# ==================================================================================================
# 3. `checklist_pass` — the verdict metric's own predicate, the plan's own three-clause definition
# (S7 spec §3.4)
# ==================================================================================================


class TestChecklistPass:
    def test_passes_when_must_contain_present_must_not_contain_absent_no_abstain(self):
        result = grounding.checklist_pass(
            "The Widget costs 24.99 and ships same day.",
            must_contain=["24.99"], must_not_contain=["19.99"], must_abstain=False,
        )
        assert result is True

    def test_fails_when_a_required_must_contain_substring_is_missing(self):
        result = grounding.checklist_pass(
            "The Widget ships same day.",
            must_contain=["24.99"], must_not_contain=["19.99"], must_abstain=False,
        )
        assert result is False

    def test_fails_when_a_forbidden_must_not_contain_substring_is_present(self):
        result = grounding.checklist_pass(
            "The Widget costs 24.99, previously 19.99.",
            must_contain=["24.99"], must_not_contain=["19.99"], must_abstain=False,
        )
        assert result is False

    def test_passes_on_a_correct_abstention_against_a_must_abstain_item(self):
        result = grounding.checklist_pass(
            "I don't know — the passages don't mention that.",
            must_contain=[], must_not_contain=[], must_abstain=True,
        )
        assert result is True

    def test_fails_when_the_reply_incorrectly_abstains_on_a_must_abstain_false_item(self):
        """Even though `mustContain`/`mustNotContain` would otherwise both pass (both lists are
        empty), abstaining when `mustAbstain` is `False` still fails the checklist — the third
        clause is independent, not a tiebreaker only consulted when the first two are
        inconclusive."""
        result = grounding.checklist_pass(
            "I don't know.", must_contain=[], must_not_contain=[], must_abstain=False,
        )
        assert result is False

    def test_fails_when_the_reply_should_abstain_but_answers_instead(self):
        result = grounding.checklist_pass(
            "The Widget costs 24.99.",
            must_contain=[], must_not_contain=[], must_abstain=True,
        )
        assert result is False


# ==================================================================================================
# 4. `format_checks` — three independent, never-pooled constraints (S7 spec §3.4). Each of the
# three fixtures below violates exactly ONE axis while staying green on the other two — the
# mutation-pair discipline `model-bench/AGENTS.md`'s guard-reach convention names (S7 spec §5 Step
# 0's own text): a passing baseline reply, and three variants each moving exactly one check.
# ==================================================================================================


_FMT = {"maxWords": 5, "mustBeSingleParagraph": True, "forbiddenPatterns": (r"^\s*[-*]\s", "```")}


class TestFormatChecks:
    def test_a_conforming_reply_passes_all_three_checks(self):
        checks = grounding.format_checks("This is a short reply.", _FMT)
        assert checks == {
            "maxWords": True, "mustBeSingleParagraph": True, "forbiddenPatterns": True,
        }

    def test_a_too_long_reply_fails_only_max_words(self):
        """5 words allowed; this reply is 8 — the other two axes stay green (no blank line, no
        forbidden pattern), proving `maxWords` moves independently."""
        checks = grounding.format_checks("This is a way too long reply here.", _FMT)
        assert checks["maxWords"] is False
        assert checks["mustBeSingleParagraph"] is True
        assert checks["forbiddenPatterns"] is True

    def test_a_two_paragraph_reply_fails_only_single_paragraph(self):
        """Exactly 5 words total (within budget), no forbidden pattern — only the blank line
        moves `mustBeSingleParagraph`."""
        checks = grounding.format_checks("This is short.\n\nSecond paragraph.", _FMT)
        assert checks["maxWords"] is True
        assert checks["mustBeSingleParagraph"] is False
        assert checks["forbiddenPatterns"] is True

    def test_a_bulleted_reply_fails_only_forbidden_patterns(self):
        """4 words, single paragraph, but the line opens with a bullet marker — only
        `forbiddenPatterns` moves."""
        checks = grounding.format_checks("- bullet point here", _FMT)
        assert checks["maxWords"] is True
        assert checks["mustBeSingleParagraph"] is True
        assert checks["forbiddenPatterns"] is False

    def test_max_words_none_never_refuses(self):
        fmt = {**_FMT, "maxWords": None}
        checks = grounding.format_checks("word " * 500, fmt)
        assert checks["maxWords"] is True

    def test_forbidden_pattern_is_matched_per_line_not_per_whole_string(self):
        """`re.MULTILINE` is what makes `^` anchor per line — the plan's own example pattern
        (`"^\\s*[-*]\\s"`) is meant to catch a bullet on ANY line, not only the first."""
        checks = grounding.format_checks(
            "Intro line.\n- hidden bullet on line two", _FMT,
        )
        assert checks["forbiddenPatterns"] is False

    def test_code_fence_pattern_is_detected(self):
        checks = grounding.format_checks("Here:\n```\ncode\n```", _FMT)
        assert checks["forbiddenPatterns"] is False


# ==================================================================================================
# 5. `build_messages` — the `ItemScorer` hook (S7 spec §2.3/§3.4). Never reads `mustContain`/
# `mustNotContain`/`mustAbstain`/`provenance` — those would leak the answer key into the prompt.
# ==================================================================================================


@pytest.fixture()
def fake_pack(tmp_path: Path) -> Pack:
    """A minimal, self-contained `chat-responder` pack: `pack.json` per S7 spec §3.2 (trimmed to
    what `build_messages` reads) plus `prompts/system.md`."""
    root = tmp_path / "chat-responder-fixture"
    (root / "prompts").mkdir(parents=True)
    (root / "prompts" / "system.md").write_text(
        "You answer questions using only the passages you are given.", encoding="utf-8",
    )
    (root / "items.jsonl").write_text("", encoding="utf-8")
    (root / "pack.json").write_text(
        """{
        "packId": "chat-responder-grounded-answers-test", "packVersion": "0.1.0",
        "role": "chat-responder", "scorer": "grounding",
        "environment": {"requires": ["lmstudio-chat"]},
        "prompt": {"systemPrompt": "prompts/system.md", "temperature": 0.0, "maxTokens": 512},
        "data": {"items": "items.jsonl"},
        "format": {"maxWords": 150, "mustBeSingleParagraph": true,
                    "forbiddenPatterns": ["^\\\\s*[-*]\\\\s", "```"]},
        "sampling": {"seed": 1, "pairingKey": ["itemId"], "analysisUnit": "itemId"},
        "metrics": {"verdictMetrics": ["groundingRate"], "headlineMetric": "groundingRate"}
        }""",
        encoding="utf-8",
    )
    return load_pack(root)


def _grounded_item(**overrides) -> dict:
    base = {
        "itemId": "cr-01", "question": "How much does the Widget cost?",
        "context": ["The Widget costs 24.99.", "The Gadget costs 19.99."],
        "mustContain": ["24.99"], "mustNotContain": ["19.99"], "mustAbstain": False,
        "provenance": {"draftedBy": "SECRET-DRAFTER", "verifiedBy": "SECRET-VERIFIER"},
    }
    base.update(overrides)
    return base


class TestBuildMessages:
    def test_system_message_starts_with_the_prompt_files_content(self, fake_pack: Pack):
        messages = grounding.build_messages(_grounded_item(), pack=fake_pack)
        expected_base = (fake_pack.root / "prompts" / "system.md").read_text(encoding="utf-8")
        assert messages[0]["role"] == "system"
        assert messages[0]["content"].startswith(expected_base)

    def test_user_message_numbers_and_labels_the_context_passages(self, fake_pack: Pack):
        messages = grounding.build_messages(_grounded_item(), pack=fake_pack)
        assert "[1] The Widget costs 24.99." in messages[1]["content"]
        assert "[2] The Gadget costs 19.99." in messages[1]["content"]

    def test_user_message_includes_the_question(self, fake_pack: Pack):
        messages = grounding.build_messages(_grounded_item(), pack=fake_pack)
        assert "QUESTION: How much does the Widget cost?" in messages[1]["content"]

    def test_no_context_renders_a_none_provided_marker(self, fake_pack: Pack):
        messages = grounding.build_messages(_grounded_item(context=[]), pack=fake_pack)
        assert "CONTEXT: (none provided)" in messages[1]["content"]

    def test_never_leaks_the_answer_key_fields_anywhere_in_the_messages(self, fake_pack: Pack):
        """The load-bearing assertion (S7 spec §2.3): none of `mustContain`/`mustNotContain`/
        `mustAbstain`/`provenance`'s own values may appear anywhere in the rendered messages."""
        item_input = _grounded_item(
            mustContain=["UNIQUE-CONTAIN-TOKEN"], mustNotContain=["UNIQUE-FORBID-TOKEN"],
        )
        messages = grounding.build_messages(item_input, pack=fake_pack)
        rendered = " ".join(m["content"] for m in messages)
        assert "UNIQUE-CONTAIN-TOKEN" not in rendered
        assert "UNIQUE-FORBID-TOKEN" not in rendered
        assert "SECRET-DRAFTER" not in rendered
        assert "SECRET-VERIFIER" not in rendered

    def test_never_raises_when_the_answer_key_fields_are_entirely_absent(self, fake_pack: Pack):
        item_input = {
            "itemId": "cr-02", "question": "Any context here?", "context": [],
        }
        messages = grounding.build_messages(item_input, pack=fake_pack)  # must not raise
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_format_directive_names_all_three_resolved_constraints(self, fake_pack: Pack):
        """`fake_pack`'s own manifest `format` block declares all three constraints
        (`maxWords: 150`, `mustBeSingleParagraph: true`, two `forbiddenPatterns`) — the resolved
        sentence `_format_directive` appends to the system prompt must name all three (code-gate
        review's own minor finding, `docs/reviews/small-model-benchmarking-s7.md`)."""
        messages = grounding.build_messages(_grounded_item(), pack=fake_pack)
        content = messages[0]["content"]
        assert "stay under 150 words" in content
        assert "write a single paragraph" in content
        assert "do not use bullet points, numbered lists, or code fences" in content

    def test_format_directive_states_no_constraints_apply_when_all_three_are_absent(
        self, fake_pack: Pack
    ):
        item_input = _grounded_item(
            format={"maxWords": None, "mustBeSingleParagraph": False, "forbiddenPatterns": []}
        )
        messages = grounding.build_messages(item_input, pack=fake_pack)
        assert "For this reply: no additional format constraints apply." in messages[0]["content"]


# ==================================================================================================
# 6. `score_item` — `result is None` branch mirrors `classification.py`'s/`extraction.py`'s own
# precedent (S7 spec §3.4); the otherwise branch scores the checklist + format checks.
# ==================================================================================================


def _chat_result(content: str) -> ChatResult:
    return ChatResult(
        message={"role": "assistant", "content": content},
        tool_calls=(), toolCallForm="prose", stats=None, model_info=None, runtime=None,
        usage=None, wallClockMs=5.0,
    )


def _timing() -> ItemTiming:
    return ItemTiming(wallClockMs=5.0, calls=(), withheldFor=None)


def _timeout_timing() -> ItemTiming:
    return ItemTiming(wallClockMs=None, calls=(), withheldFor="timeout")


def _no_response_timing() -> ItemTiming:
    return ItemTiming(wallClockMs=None, calls=(), withheldFor="no_response")


class TestScoreItem:
    def test_timeout_scores_fail_with_no_scoreable_metrics(self, fake_pack: Pack):
        result = grounding.score_item(
            _grounded_item(), None, _timeout_timing(), pack=fake_pack
        )
        assert result.outcome == "fail"
        assert result.scoreable == {}
        assert result.counts == {}

    def test_no_response_scores_unrunnable_with_no_scoreable_metrics(self, fake_pack: Pack):
        result = grounding.score_item(
            _grounded_item(), None, _no_response_timing(), pack=fake_pack
        )
        assert result.outcome == "unrunnable"
        assert result.scoreable == {}
        assert result.counts == {}

    def test_a_clean_grounded_reply_scores_pass_with_all_four_metrics_declared(
        self, fake_pack: Pack
    ):
        reply = _chat_result("The Widget costs 24.99.")
        result = grounding.score_item(_grounded_item(), reply, _timing(), pack=fake_pack)
        assert result.outcome == "pass"
        assert result.scoreable == {
            "groundingRate": True, "formatMaxWords": True,
            "formatSingleParagraph": True, "formatNoForbiddenPatterns": True,
        }
        assert result.counts == {
            "groundingRate": 1, "formatMaxWords": 1,
            "formatSingleParagraph": 1, "formatNoForbiddenPatterns": 1,
        }

    def test_a_reply_missing_a_required_substring_scores_zero_on_grounding_rate(
        self, fake_pack: Pack
    ):
        reply = _chat_result("The Widget ships same day.")
        result = grounding.score_item(_grounded_item(), reply, _timing(), pack=fake_pack)
        assert result.outcome == "pass"
        assert result.counts["groundingRate"] == 0

    def test_a_non_string_reply_content_is_treated_as_empty_never_raises(self, fake_pack: Pack):
        reply = _chat_result("")
        reply.message["content"] = None
        result = grounding.score_item(_grounded_item(), reply, _timing(), pack=fake_pack)
        assert result.outcome == "pass"
        assert result.counts["groundingRate"] == 0


# ==================================================================================================
# 7. `aggregate` — four independent `BinaryMetric`s, each over only the items that declared it
# (trivially all of them, since `score_item` always declares all four) — S7 spec §3.4.
# ==================================================================================================


_FILLER_200_WORDS = " ".join(f"filler{i}" for i in range(200))


def _scored(item_id: str, *, contain_ok: bool, format_ok: bool, pack: Pack) -> ItemResult:
    """Routes through the real `score_item`, never a hand-built `ItemResult` — mirrors
    `test_scoring_classification.py`'s own `_scored` helper precedent. `contain_ok` controls
    whether `groundingRate` succeeds; `format_ok` controls all three format checks together — the
    "bad" content opens with a bullet marker (forbiddenPatterns), carries a blank line (two
    paragraphs) and 200+ filler words (over the fixture pack's 150-word budget), breaking all
    three format axes at once so the pooled `n`/`successes` in the aggregate tests are simple to
    reason about."""
    item_input = _grounded_item(itemId=item_id, mustContain=["24.99"], mustNotContain=["19.99"])
    fact = "24.99" if contain_ok else "unclear"
    if format_ok:
        content = f"It is {fact}."
    else:
        content = f"- It is {fact}.\n\n{_FILLER_200_WORDS}"
    return grounding.score_item(item_input, _chat_result(content), _timing(), pack=pack)


class TestAggregate:
    def test_grounding_rate_pools_every_items_own_checklist_outcome(self, fake_pack: Pack):
        items = [
            _scored("a", contain_ok=True, format_ok=True, pack=fake_pack),
            _scored("b", contain_ok=True, format_ok=True, pack=fake_pack),
            _scored("c", contain_ok=False, format_ok=True, pack=fake_pack),
        ]
        aggregates = grounding.aggregate(items, pack=fake_pack)
        assert aggregates.checklistPass.name == "groundingRate"
        assert aggregates.checklistPass.n == 3
        assert aggregates.checklistPass.successes == 2

    def test_the_three_format_checks_are_independent_binary_metrics(self, fake_pack: Pack):
        items = [
            _scored("a", contain_ok=True, format_ok=True, pack=fake_pack),
            _scored("b", contain_ok=True, format_ok=False, pack=fake_pack),
        ]
        aggregates = grounding.aggregate(items, pack=fake_pack)
        by_name = {m.name: m for m in aggregates.perCheck}
        assert set(by_name) == {
            "formatMaxWords", "formatSingleParagraph", "formatNoForbiddenPatterns",
        }
        for metric in by_name.values():
            assert metric.n == 2
            assert metric.successes == 1  # only item "a" respects every format constraint

    def test_every_binary_metric_uses_the_item_unit(self, fake_pack: Pack):
        items = [_scored("a", contain_ok=True, format_ok=True, pack=fake_pack)]
        aggregates = grounding.aggregate(items, pack=fake_pack)
        assert aggregates.checklistPass.unit == "item"
        assert all(m.unit == "item" for m in aggregates.perCheck)

    def test_parse_failures_is_always_zero_for_this_scorer(self, fake_pack: Pack):
        items = [_scored("a", contain_ok=True, format_ok=True, pack=fake_pack)]
        aggregates = grounding.aggregate(items, pack=fake_pack)
        assert aggregates.parseFailures == 0

    def test_an_n_never_special_cases_one_metric_over_another(self, fake_pack: Pack):
        """A future edit could accidentally special-case one metric's denominator — pinned by
        asserting all four share the same `n` when every item declares all four (score_item's own
        invariant)."""
        items = [
            _scored("a", contain_ok=True, format_ok=True, pack=fake_pack),
            _scored("b", contain_ok=False, format_ok=False, pack=fake_pack),
            _scored("c", contain_ok=True, format_ok=False, pack=fake_pack),
        ]
        aggregates = grounding.aggregate(items, pack=fake_pack)
        ns = {aggregates.checklistPass.n, *[m.n for m in aggregates.perCheck]}
        assert ns == {3}

    def test_an_unrunnable_item_is_excluded_from_every_metrics_denominator(self, fake_pack: Pack):
        """`score_item`'s `result is None` branch returns `scoreable={}` (S7 spec §3.4) — this
        pins that `aggregate` actually honors that, not just that `score_item` produces it: an
        unrunnable item must not appear in any metric's `n`, the exact "precondition failure
        never silently out of the denominator" property `ItemResult`'s own docstring states."""
        unrunnable = grounding.score_item(
            _grounded_item(), None, _no_response_timing(), pack=fake_pack
        )
        items = [
            _scored("a", contain_ok=True, format_ok=True, pack=fake_pack),
            _scored("b", contain_ok=False, format_ok=True, pack=fake_pack),
            unrunnable,
        ]
        aggregates = grounding.aggregate(items, pack=fake_pack)
        assert aggregates.checklistPass.n == 2
        assert aggregates.checklistPass.successes == 1
        for metric in aggregates.perCheck:
            assert metric.n == 2
