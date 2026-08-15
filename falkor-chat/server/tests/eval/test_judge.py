"""Unit tests for the GraphRAG judge layer (K-026, `docs/plans/graphrag-eval.md` §5
Unit 3): `judge.py`'s prompt builder + conservative-on-ambiguity parser.

Genuinely network/DB-free (no marker, part of the default `pytest -q` run) — the judge
LLM is a scripted stub, exactly the `guards.py`/`test_guards.py` precedent this module
mirrors. The live counterpart, `test_judge_live.py` (`pytest.mark.live`), drives the
same `judge_triple` against a real model.
"""

from __future__ import annotations

import pytest

from judge import JudgeVerdict, build_judge_prompt, judge_triple


class StubJudgeLLM:
    """A scripted judge `.complete(...)` returning a fixed raw string reply."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages):
        self.calls.append(messages)
        return self.reply


# ── build_judge_prompt ───────────────────────────────────────────────────────


def test_prompt_renders_context_as_bullets():
    messages = build_judge_prompt(
        "why did it break?", ["seed one", "seed two"], "because of X"
    )
    user = messages[-1]["content"]
    assert "- seed one" in user
    assert "- seed two" in user
    assert "why did it break?" in user
    assert "because of X" in user


def test_prompt_marks_empty_context_explicitly():
    """Empty context must render as an explicit marker, not a blank/absent block —
    the judge needs to be able to tell 'no context' apart from 'context omitted'."""
    messages = build_judge_prompt("q", [], "a")
    user = messages[-1]["content"]
    assert "none" in user.lower()


# ── judge_triple: clean verdicts ─────────────────────────────────────────────


def test_judge_triple_parses_clean_true_true_verdict():
    llm = StubJudgeLLM('{"faithfulness": true, "relevance": true, "rationale": "grounded"}')
    verdict = judge_triple(llm, question="q", context=["ctx"], answer="a")
    assert verdict == JudgeVerdict(faithfulness=True, relevance=True, rationale="grounded")
    assert verdict.parse_failed is False


def test_judge_triple_parses_clean_false_false_verdict():
    llm = StubJudgeLLM(
        '{"faithfulness": false, "relevance": false, "rationale": "contradicts context"}'
    )
    verdict = judge_triple(llm, question="q", context=["ctx"], answer="a")
    assert verdict.faithfulness is False
    assert verdict.relevance is False
    assert verdict.parse_failed is False


def test_judge_triple_accepts_fenced_json():
    llm = StubJudgeLLM('```json\n{"faithfulness": true, "relevance": false}\n```')
    verdict = judge_triple(llm, question="q", context=["ctx"], answer="a")
    assert verdict.faithfulness is True
    assert verdict.relevance is False
    assert verdict.parse_failed is False


# ── judge_triple: empty-context abstain override (plan §6 edge case) ────────


def test_empty_context_forces_faithfulness_none_even_if_judge_said_true():
    """A judge that (wrongly) scores faithfulness against no context must be
    overridden — faithfulness is never scored against nothing, unconditionally."""
    llm = StubJudgeLLM('{"faithfulness": true, "relevance": true, "rationale": "fine"}')
    verdict = judge_triple(llm, question="q", context=[], answer="a")
    assert verdict.faithfulness is None
    assert verdict.relevance is True  # relevance is unaffected by the override
    assert verdict.parse_failed is False


def test_empty_context_forces_faithfulness_none_even_if_judge_said_false():
    llm = StubJudgeLLM('{"faithfulness": false, "relevance": true}')
    verdict = judge_triple(llm, question="q", context=[], answer="a")
    assert verdict.faithfulness is None


def test_empty_context_with_judge_correctly_abstaining_stays_none():
    llm = StubJudgeLLM('{"faithfulness": null, "relevance": true}')
    verdict = judge_triple(llm, question="q", context=[], answer="a")
    assert verdict.faithfulness is None
    assert verdict.parse_failed is False


def test_nonempty_context_is_not_overridden():
    llm = StubJudgeLLM('{"faithfulness": true, "relevance": true}')
    verdict = judge_triple(llm, question="q", context=["some real context"], answer="a")
    assert verdict.faithfulness is True


# ── judge_triple: conservative-on-ambiguity parsing (mutation-test target) ──


@pytest.mark.parametrize(
    "bad_reply",
    [
        "not json at all, just prose",
        '{"faithfulness": true, "relevance": true',  # truncated/malformed JSON
        # quoted mid-sentence
        'I would answer {"faithfulness": true, "relevance": true} but unsure.',
        "",
        "   ",
    ],
)
def test_unparseable_reply_resolves_conservative_and_flags_parse_failure(bad_reply):
    llm = StubJudgeLLM(bad_reply)
    verdict = judge_triple(llm, question="q", context=["ctx"], answer="a")
    assert verdict.faithfulness is None
    assert verdict.relevance is False
    assert verdict.parse_failed is True


def test_two_ambiguous_candidate_objects_resolve_conservative():
    """extract_own_line_json_object refuses to guess between two own-line objects —
    judge_triple must inherit that refusal rather than picking either one."""
    reply = (
        '{"faithfulness": true, "relevance": true}\n'
        '{"faithfulness": false, "relevance": false}'
    )
    llm = StubJudgeLLM(reply)
    verdict = judge_triple(llm, question="q", context=["ctx"], answer="a")
    assert verdict.parse_failed is True
    assert verdict.faithfulness is None
    assert verdict.relevance is False


def test_whole_reply_missing_relevance_key_is_not_a_parse_failure():
    """`extract_own_line_json_object` accepts a reply that IS ENTIRELY one JSON
    object unconditionally (require_key only disambiguates the own-line-scan
    branch, per its own docstring) — so a whole-reply object missing `relevance`
    still parses; `_coerce_verdict` is what conservatively defaults the missing
    axis to `False`, not a parse failure."""
    llm = StubJudgeLLM('{"faithfulness": true}')
    verdict = judge_triple(llm, question="q", context=["ctx"], answer="a")
    assert verdict.parse_failed is False
    assert verdict.relevance is False  # missing key -> conservative False
    assert verdict.faithfulness is True


def test_own_line_object_missing_relevance_key_is_filtered_by_require_key():
    """When the verdict is NOT the whole reply (prose around it), require_key
    does its job: an own-line object lacking `relevance` is not a candidate at
    all, so with no other candidate the parse resolves conservative."""
    reply = 'Let me think about this.\n{"faithfulness": true}\nThat is my analysis.'
    llm = StubJudgeLLM(reply)
    verdict = judge_triple(llm, question="q", context=["ctx"], answer="a")
    assert verdict.parse_failed is True
    assert verdict.faithfulness is None
    assert verdict.relevance is False


def test_require_key_disambiguates_among_multiple_own_line_objects():
    """Two own-line objects, only one carrying `relevance` — require_key narrows
    to exactly that one rather than the ambiguous-multiple-matches path."""
    reply = (
        "Notes:\n"
        '{"faithfulness": true}\n'
        '{"faithfulness": true, "relevance": true}\n'
        "done"
    )
    llm = StubJudgeLLM(reply)
    verdict = judge_triple(llm, question="q", context=["ctx"], answer="a")
    assert verdict.parse_failed is False
    assert verdict.faithfulness is True
    assert verdict.relevance is True


def test_non_bool_relevance_resolves_false_but_not_a_parse_failure():
    """The object parses fine (carries the required key) but relevance's value is
    not a clean bool — coerced to False, not counted as a parse failure."""
    llm = StubJudgeLLM('{"faithfulness": true, "relevance": "yes"}')
    verdict = judge_triple(llm, question="q", context=["ctx"], answer="a")
    assert verdict.parse_failed is False
    assert verdict.relevance is False


def test_non_bool_faithfulness_resolves_none_but_not_a_parse_failure():
    llm = StubJudgeLLM('{"faithfulness": "yes", "relevance": true}')
    verdict = judge_triple(llm, question="q", context=["ctx"], answer="a")
    assert verdict.parse_failed is False
    assert verdict.faithfulness is None
