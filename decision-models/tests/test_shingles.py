"""`decisionmodels/shingles.py` (typed-decision-models U5.10a; note §Q1.3 "Shingle check", §Q1.4;
plan2 §3.5). Expected values are written here, never read from the constants they check."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from decisionmodels import shingles

EIGHT = "one two three four five six seven eight"


def test_tokens_are_nfkc_lowercase_letters_and_digits_split_on_everything_else() -> None:
    assert shingles.tokens("Olá, MUNDO! 3x-4") == ["olá", "mundo", "3x", "4"]
    assert shingles.tokens("ＡＢＣ １２３") == ["abc", "123"]       # NFKC folds full-width forms
    assert shingles.tokens("é") == ["é"]                     # and composes
    assert shingles.tokens("snake_case.and/slash") == ["snake", "case", "and", "slash"]
    assert shingles.tokens("") == [] and shingles.tokens("---") == []


def test_a_shingle_is_eight_consecutive_tokens_and_shorter_text_has_none() -> None:
    assert shingles.shingles("a b c d e f g") == set()
    assert shingles.shingles(EIGHT) == {tuple(EIGHT.split())}
    assert shingles.shingles(EIGHT + " nine") == {
        tuple("one two three four five six seven eight".split()),
        tuple("two three four five six seven eight nine".split())}


def test_the_hash_is_the_sha256_of_the_tokens_joined_by_one_space() -> None:
    assert shingles.shingle_hash(EIGHT.split()) == hashlib.sha256(EIGHT.encode()).hexdigest()
    assert shingles.shingle_hashes(EIGHT) == {hashlib.sha256(EIGHT.encode()).hexdigest()}


def test_evidence_text_is_the_rendered_evidence_fields_only() -> None:
    guard = {"condition": "CONDSENT", "path": "turns", "understanding": {
        "request": "REQ", "known": ["K1", "K2"], "missing": ["M1"]},
        "turns": [{"role": "user", "displayName": "SPEAKER", "text": "T1"},
                  {"role": "assistant", "text": "T2"}]}
    assert shingles.evidence_text("guard-judge", guard) == "K1\nK2\nM1\nREQ\nT1\nT2"
    assert shingles.evidence_text("guard-judge", {"condition": "c", "turns": []}) == ""
    join_in = {"agentName": "AGENT", "priorTurns": [{"displayName": "S", "text": "P1"}],
               "newMessage": {"displayName": "S2", "text": "NEW"}}
    assert shingles.evidence_text("join-in-judge", join_in) == "P1\nNEW"
    message = {"text": "MSG", "priorTurn": {"speaker": "SP", "text": "PRIOR"}}
    assert shingles.evidence_text("message-type-classifier", message) == "PRIOR\nMSG"
    assert shingles.evidence_text("message-type-classifier",
                                  {"text": "MSG", "priorTurn": None}) == "MSG"


def test_a_condition_sentence_shared_by_two_items_gives_no_shared_shingle() -> None:
    condition = "the user has provided enough information to research their request"
    one = {"condition": condition, "understanding": {"request": "alpha beta"}, "turns": []}
    two = {"condition": condition, "understanding": {"request": "gamma delta"}, "turns": []}
    for item in (one, two):
        assert shingles.shingles(shingles.evidence_text("guard-judge", item)) == set()
    assert len(shingles.shingles(condition)) > 0          # it would have matched, were it read


def test_an_unknown_role_is_refused() -> None:
    with pytest.raises(ValueError, match="no evidence fields"):
        shingles.evidence_text("nope", {})


def test_a_paraphrase_that_keeps_an_eight_token_span_hits_the_side_check() -> None:
    side = frozenset(shingles.shingle_hashes(
        "The checkout service started returning HTTP 502 errors"))
    assert len(side) == 1
    assert shingles.hits_side_check(
        "WHY did the checkout service started RETURNING http 502 errors? Please look", side)
    assert not shingles.hits_side_check("the checkout service began returning HTTP 502 errors",
                                        side)
    assert not shingles.hits_side_check("", side) and not shingles.hits_side_check(EIGHT, set())


def test_load_side_check_hashes_skips_headers_and_blanks(tmp_path: Path) -> None:
    digest = hashlib.sha256(b"x").hexdigest()
    path = tmp_path / "side.txt"
    path.write_text(f"# source-sha256: abc\n# shingle-tokens: 8\n\n{digest}\n{digest}\n")
    assert shingles.load_side_check_hashes(path) == frozenset({digest})


@pytest.mark.parametrize("line", ["not a hash", "A" * 64, "a" * 63, "a" * 65])
def test_a_damaged_side_check_file_raises_rather_than_loading_empty(
        tmp_path: Path, line: str) -> None:
    path = tmp_path / "side.txt"
    path.write_text(f"# header\n{line}\n")
    with pytest.raises(ValueError, match="line 2 is not a sha256"):
        shingles.load_side_check_hashes(path)


def test_a_missing_side_check_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        shingles.load_side_check_hashes(tmp_path / "none.txt")


def test_the_committed_side_check_file_loads_and_names_its_source() -> None:
    path = shingles.SIDE_CHECK_FILE
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# source-sha256: ")
    assert len(shingles.load_side_check_hashes(path)) > 1000


def test_the_committed_side_check_file_is_of_the_current_side_check_items() -> None:
    items = (Path(__file__).resolve().parents[2] / "model-bench" / "packs"
             / "guard-judge-understanding" / "items.jsonl")
    if not items.is_file():
        pytest.skip("model-bench/'s side-check pack is absent from this checkout")
    first = shingles.SIDE_CHECK_FILE.read_text(encoding="utf-8").splitlines()[0]
    assert first == f"# source-sha256: {hashlib.sha256(items.read_bytes()).hexdigest()}"
