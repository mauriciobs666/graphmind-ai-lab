"""P1-26 dispatch 2 (d): the interpreter's prompt and grammar (``ccg/front/prompts/interpret.txt``, ``ccg/front/grammar/interpret.json``).

Contract: ``docs/plans/connected-code-graph-p1-ml.md`` §4.2: at most 4 few-shot examples, over fixture names ``fx-*`` only, covering English and Portuguese, both directions, a
``null`` direction, one qualified group, one unbuilt use case and one two-use-case question; the example cues share no phrase with any dev AC-24 question
(``cpg/eval/dev/ac24.jsonl``), so the dev entry check is not tuned on its own items; the grammar is the ``InterpretDraft`` of ``ccg.front.draft``. Held-out question text
appears nowhere: nothing in this repository's tests reads it, and the prompt is built from made-up ``fx-*`` names only.

    cd cpg && .venv/bin/python -m pytest tests/test_front_prompt.py
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from ccg.front import draft as D
from ccg.front import interpret as I

CPG = Path(__file__).resolve().parent.parent
DEV = CPG / "eval" / "dev" / "ac24.jsonl"
PROMPT = I.prompt_text()
EXAMPLES = re.findall(r"^Q: (.*)\nA: (.*)$", PROMPT, re.M)


def fold(text: str) -> str:
    """Lower case with accents removed, so ``relações`` and ``relacoes`` (or ``função`` and ``funcao``) compare equal."""
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def words(text: str) -> list[str]:
    return re.findall(r"\w+", fold(text))


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def dev_texts() -> list[str]:
    return [json.loads(line)["text"] for line in DEV.read_text(encoding="utf-8").splitlines()]


def test_the_files_are_txt_and_json_never_markdown():
    assert (CPG / "ccg" / "front" / "prompts" / "interpret.txt").is_file() and (CPG / "ccg" / "front" / "grammar" / "interpret.json").is_file()
    assert not [p for p in (CPG / "ccg" / "front").rglob("*") if p.suffix.lower() == ".md"]


def test_there_are_at_most_four_examples_and_each_reply_is_valid_against_the_grammar():
    assert 1 <= len(EXAMPLES) <= 4
    validator = Draft202012Validator(I.grammar())
    for q, a in EXAMPLES:
        reply = json.loads(a)
        assert not list(validator.iter_errors(reply)), q
        assert I.to_draft(reply) is not None


def test_every_example_span_is_a_verbatim_substring_of_its_question_and_names_a_fixture():
    for q, a in EXAMPLES:
        for group in json.loads(a)["anchors"]:
            for span in group["spans"]:
                assert span["text"] in q, (q, span)
                assert "fx-" in span["text"] or "fx_" in span["text"], span                     # fixture names only


def test_the_examples_cover_the_axes_the_method_note_names():
    replies = [json.loads(a) for _, a in EXAMPLES]
    assert {r["direction"] for r in replies} == {"impact", "depends-on", None}
    assert any(len(g["spans"]) >= 2 for r in replies for g in r["anchors"])                    # a qualified group
    assert any(x["useCase"] in D.UNBUILT_USE_CASES for r in replies for x in r["readings"] + [r])      # an unbuilt use case
    assert any(r["readings"] for r in replies)                                                 # a two-use-case question
    questions = " ".join(q.lower() for q, _ in EXAMPLES)
    assert any(w in words(questions) for w in ("precisa", "faca", "quem", "que")) and any(w in words(questions) for w in ("consumers", "show", "touch"))   # PT and EN


def test_the_prompt_shares_no_three_word_phrase_with_any_dev_question():
    dev = set().union(*(ngrams(t, 3) for t in dev_texts()))
    assert len(dev_texts()) == 14 and dev
    shared = ngrams(PROMPT, 3) & dev
    assert not shared, sorted(shared)


def test_the_example_questions_share_no_two_word_phrase_with_any_dev_question():
    dev = set().union(*(ngrams(t, 2) for t in dev_texts()))
    shared = set().union(*(ngrams(q, 2) for q, _ in EXAMPLES)) & dev
    assert not shared, sorted(shared)


NEUTRAL_CUES = ("an overview of", "o panorama de")          # what the prompt teaches as wording that states no direction (field text and example 3)
DEPENDS_ON_CUE = "what the anchor itself calls or uses"      # the subject-worded cue: a passive "what calls this route" is not a depends-on cue


def test_the_neutral_null_direction_cue_appears_in_no_dev_question_even_by_accent_or_case():
    dev = [" ".join(words(t)) for t in dev_texts()]
    for cue in NEUTRAL_CUES:
        folded = " ".join(words(cue))
        assert not any(w in " ".join(dev).split() for w in folded.split() if w not in ("an", "of", "o", "de")), cue      # no content word of the cue is a dev word
        assert not any(folded in t for t in dev), cue
        assert cue in PROMPT, cue
    assert PROMPT.count(NEUTRAL_CUES[0]) == 2 and PROMPT.count(NEUTRAL_CUES[1]) == 1       # field text (both cues) and example 3
    assert "relations" not in words(PROMPT) and "relacoes" not in words(PROMPT)            # the cue of dev item QW-47 is taught nowhere


def test_the_depends_on_cue_is_worded_by_subject_so_a_passive_calls_question_is_not_one():
    field = next(line for line in PROMPT.splitlines() if line.startswith("- direction:"))
    assert DEPENDS_ON_CUE in field
    assert not re.search(r"\bcalls\b(?! or uses)", field)                                  # no bare "calls" cue
    assert "Never answer null when the question implies a direction" in field


def test_accent_folding_is_what_the_overlap_checks_compare():
    assert fold("Relações da função") == "relacoes da funcao" and words("essa FUNÇÃO") == ["essa", "funcao"]


def test_no_dev_question_appears_in_the_prompt_and_no_dev_anchor_name_either():
    for t in dev_texts():
        assert t.lower() not in PROMPT.lower()
    for row in (json.loads(line) for line in DEV.read_text(encoding="utf-8").splitlines()):
        for tok in re.findall(r"[\w./{}:-]{6,}", row["text"]):
            if "/" in tok or "_" in tok or "." in tok:
                assert tok not in PROMPT, tok


def test_the_prompt_names_every_kind_hint_and_every_unbuilt_use_case_and_the_null_rule():
    for kind in D.KIND_HINTS:
        assert kind in PROMPT
    for use_case in (D.BUILT_USE_CASE, *D.UNBUILT_USE_CASES):
        assert f'"{use_case}"' in PROMPT
    assert "Never answer null when the question implies a direction" in PROMPT and "character for character" in PROMPT


def test_the_grammar_is_a_valid_schema_that_states_the_drafts_own_limits():
    g = I.grammar()
    Draft202012Validator.check_schema(g)
    props = g["properties"]
    assert props["useCase"]["enum"] == [D.BUILT_USE_CASE, *D.UNBUILT_USE_CASES]
    assert props["direction"]["enum"] == [*D.DIRECTIONS, None]
    anchors = props["anchors"]
    assert (anchors["minItems"], anchors["maxItems"]) == (1, D.MAX_GROUPS)
    spans = anchors["items"]["properties"]["spans"]
    assert (spans["minItems"], spans["maxItems"]) == (1, D.MAX_SPANS)
    assert spans["items"]["properties"]["kindHint"]["enum"] == list(D.KIND_HINTS)
    assert g["additionalProperties"] is False and sorted(g["required"]) == sorted(props)


def test_the_hashes_are_the_sha256_of_the_files():
    import hashlib
    assert I.prompt_hash() == hashlib.sha256(Path(I.HERE / "prompts" / "interpret.txt").read_bytes()).hexdigest()
    assert I.grammar_hash() == hashlib.sha256(Path(I.HERE / "grammar" / "interpret.json").read_bytes()).hexdigest()
