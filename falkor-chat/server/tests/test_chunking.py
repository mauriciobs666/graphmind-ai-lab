"""Unit tests for `chunking.split_into_chunks` — pure, no I/O, no fixtures needed.

Covers the plan §3.2 boundary-rule priority order: paragraph breaks > sentence
boundaries > hard cut, with overlap carried forward at every chunk boundary.
"""

from __future__ import annotations

from falkorchat.chunking import split_into_chunks


def test_short_text_is_one_chunk():
    assert split_into_chunks("hello world") == ["hello world"]


def test_empty_text_yields_no_chunks():
    assert split_into_chunks("") == []


def test_whitespace_only_text_yields_no_chunks():
    assert split_into_chunks("   \n\n\t  ") == []


def test_exact_boundary_text_is_one_chunk():
    text = "x" * 1000
    chunks = split_into_chunks(text, size=1000, overlap=150)
    assert chunks == [text]


def test_one_char_over_boundary_hard_cuts_and_carries_overlap():
    text = "x" * 1001
    chunks = split_into_chunks(text, size=1000, overlap=150)
    assert len(chunks) == 2
    assert len(chunks[0]) == 1000
    # second chunk = last 150 chars of chunk 1 (overlap) + the 1 remaining char
    assert len(chunks[1]) == 151
    assert chunks[1] == chunks[0][-150:] + "x"


def test_paragraph_breaks_preferred_when_everything_fits():
    text = "para one.\n\npara two.\n\npara three."
    chunks = split_into_chunks(text, size=1000, overlap=150)
    # small enough that all three paragraphs land in a single chunk, verbatim
    assert chunks == [text]


def test_long_paragraph_splits_on_sentence_boundaries():
    # One paragraph (no \n\n), built from 100 short sentences — long enough to
    # need multiple chunks, but every individual sentence is well under `size`,
    # so this exercises the sentence-boundary path, not the hard-cut path.
    sentence = "This is a sentence. "
    text = sentence * 100  # 2100 chars
    chunks = split_into_chunks(text, size=1000, overlap=150)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 1000 + 150  # target + at most one overlap carry
    # every chunk boundary lands on a sentence boundary, not mid-sentence —
    # each chunk (after stripping the leading overlap carry-over) is made of
    # whole ". "-terminated sentences.
    for chunk in chunks:
        assert chunk.endswith(". ") or chunk.endswith(".")
    # reconstructing the original text from the chunks (each chunk after the
    # first has its leading `overlap` chars stripped) round-trips exactly —
    # proves no content was dropped or duplicated beyond the overlap.
    reconstructed = chunks[0] + "".join(c[150:] for c in chunks[1:])
    assert reconstructed == text


def test_long_single_sentence_hard_cuts():
    # No paragraph break, no sentence delimiter anywhere — a single "sentence"
    # (e.g. a large embedded JSON/CSV blob per the plan's real-world cases).
    blob = "A" * 3000
    chunks = split_into_chunks(blob, size=1000, overlap=150)

    assert len(chunks) == 3
    assert len(chunks[0]) == 1000
    # chunk 2/3 = 150-char overlap carry + 1000 fresh chars = 1150
    assert len(chunks[1]) == 1150
    assert len(chunks[2]) == 1150
    assert all(c == "A" * len(c) for c in chunks)


def test_mixed_paragraphs_one_oversized_one_not():
    short_para = "Short intro paragraph.\n\n"
    long_para = "Sentence one. " * 100  # ~1500 chars, one paragraph, no \n\n
    text = short_para + long_para
    chunks = split_into_chunks(text, size=1000, overlap=150)

    assert len(chunks) >= 2
    # the short paragraph survives intact at the start of the first chunk
    assert chunks[0].startswith("Short intro paragraph.")


def test_default_size_and_overlap_match_plan_recommendation():
    text = "x" * 2500
    chunks = split_into_chunks(text)  # size=1000, overlap=150 defaults
    assert len(chunks) == 3
    assert len(chunks[0]) == 1000
