"""Unit tests for `update_detection.py` (document-ingestion2 Stage D, FR-2).

Every function here is pure (no I/O, no Cypher — mirrors `test_fusion.py`'s
posture) so the whole module is cheaply testable in full: `normalize_text`,
`content_hash`, `shingles`, `jaccard` (the ML note's own named cases — empty
text, whitespace-only variance, identical-modulo-case-and-whitespace pairs,
a hard-negative pair sharing heavy boilerplate) plus `minhash_signature`/
`lsh_bands` (graph-dba's candidate-generation redesign, plan §0/§7).
"""

from __future__ import annotations

from falkorchat import update_detection as ud

# ── normalize_text ────────────────────────────────────────────────────────────


def test_normalize_text_collapses_whitespace_runs():
    assert ud.normalize_text("hello    world\n\tagain") == "hello world again"


def test_normalize_text_strips_leading_and_trailing_whitespace():
    assert ud.normalize_text("   hello world   ") == "hello world"


def test_normalize_text_case_folds():
    assert ud.normalize_text("Hello WORLD") == "hello world"


def test_normalize_text_empty_string_is_empty():
    assert ud.normalize_text("") == ""


def test_normalize_text_whitespace_only_is_empty():
    assert ud.normalize_text("   \n\t  ") == ""


def test_normalize_text_makes_case_and_whitespace_variants_identical():
    a = ud.normalize_text("Hello   World")
    b = ud.normalize_text("hello world")
    c = ud.normalize_text("  HELLO\nWORLD  ")
    assert a == b == c


# ── content_hash ──────────────────────────────────────────────────────────────


def test_content_hash_is_deterministic():
    normalized = ud.normalize_text("Some Document Text")
    assert ud.content_hash(normalized) == ud.content_hash(normalized)


def test_content_hash_matches_for_case_and_whitespace_variants():
    h1 = ud.content_hash(ud.normalize_text("Hello   World"))
    h2 = ud.content_hash(ud.normalize_text("  hello\nworld  "))
    assert h1 == h2


def test_content_hash_differs_for_different_content():
    h1 = ud.content_hash(ud.normalize_text("Hello World"))
    h2 = ud.content_hash(ud.normalize_text("Goodbye World"))
    assert h1 != h2


def test_content_hash_of_empty_text_is_stable_and_well_defined():
    h1 = ud.content_hash(ud.normalize_text(""))
    h2 = ud.content_hash(ud.normalize_text("   "))
    assert h1 == h2  # both normalize to "" first
    assert isinstance(h1, str) and len(h1) == 64  # sha256 hex digest


# ── shingles ───────────────────────────────────────────────────────────────────


def test_shingles_of_empty_text_is_empty_set():
    assert ud.shingles(ud.normalize_text("")) == set()


def test_shingles_of_whitespace_only_text_is_empty_set():
    assert ud.shingles(ud.normalize_text("   \n\t  ")) == set()


def test_shingles_shorter_than_n_yields_one_shingle_of_the_whole_text():
    normalized = ud.normalize_text("one two three")
    result = ud.shingles(normalized, n=5)
    assert result == {"one two three"}


def test_shingles_builds_5_word_ngrams_by_default():
    normalized = ud.normalize_text("a b c d e f")
    result = ud.shingles(normalized)
    assert result == {"a b c d e", "b c d e f"}


def test_shingles_of_case_and_whitespace_variants_are_identical():
    a = ud.shingles(ud.normalize_text("The Quick Brown Fox Jumps Over"))
    b = ud.shingles(ud.normalize_text("the   quick\nbrown fox jumps over"))
    assert a == b


# ── jaccard ────────────────────────────────────────────────────────────────────


def test_jaccard_of_identical_sets_is_one():
    s = ud.shingles(ud.normalize_text("a b c d e f g"))
    assert ud.jaccard(s, s) == 1.0


def test_jaccard_of_disjoint_sets_is_zero():
    a = ud.shingles(ud.normalize_text("alpha beta gamma delta epsilon"))
    b = ud.shingles(ud.normalize_text("zulu yankee xray whiskey victor"))
    assert ud.jaccard(a, b) == 0.0


def test_jaccard_of_two_empty_sets_is_zero_not_one():
    # No shared content to measure an overlap OF — identity is content_hash's
    # job (the auto tier), not this function's (ML note §3.2/§5).
    assert ud.jaccard(set(), set()) == 0.0


def test_jaccard_partial_overlap_is_between_zero_and_one():
    a = ud.shingles(ud.normalize_text("one two three four five six"))
    b = ud.shingles(ud.normalize_text("one two three four five seven"))
    ratio = ud.jaccard(a, b)
    assert 0.0 < ratio < 1.0


def test_jaccard_hard_negative_shared_boilerplate_is_high_but_documents_differ():
    """ML note §6's named hard-negative class: two genuinely different
    documents built from the same heavy-boilerplate template. Jaccard alone
    cannot distinguish this from a real edit — that is the accepted v1 limit
    the ML note names, not a bug this test is asserting against; it only
    pins that the ratio is high (as expected for shared boilerplate), not
    that this function somehow knows the documents are unrelated.
    """
    boilerplate = " ".join(f"word{i}" for i in range(50))
    doc_a = ud.shingles(ud.normalize_text(f"{boilerplate} unique-a"))
    doc_b = ud.shingles(ud.normalize_text(f"{boilerplate} unique-b"))
    ratio = ud.jaccard(doc_a, doc_b)
    assert ratio > 0.8  # dominated by shared boilerplate
    assert ratio < 1.0  # but not identical


# ── minhash_signature ───────────────────────────────────────────────────────────


def test_minhash_signature_is_deterministic_across_calls():
    s = ud.shingles(ud.normalize_text("the quick brown fox jumps over the lazy dog"))
    assert ud.minhash_signature(s) == ud.minhash_signature(s)


def test_minhash_signature_has_length_k():
    s = ud.shingles(ud.normalize_text("one two three four five six seven"))
    assert len(ud.minhash_signature(s, k=16)) == 16
    assert len(ud.minhash_signature(s, k=32)) == 32


def test_minhash_signature_of_empty_shingles_is_all_zeros():
    assert ud.minhash_signature(set(), k=8) == [0] * 8


def test_minhash_signature_identical_shingle_sets_produce_identical_signature():
    a = ud.shingles(ud.normalize_text("Hello   World Foo Bar Baz"))
    b = ud.shingles(ud.normalize_text("hello world\nfoo bar baz"))
    assert ud.minhash_signature(a) == ud.minhash_signature(b)


def test_minhash_signature_differs_for_disjoint_shingle_sets():
    a = ud.shingles(ud.normalize_text("alpha beta gamma delta epsilon zeta"))
    b = ud.shingles(ud.normalize_text("zulu yankee xray whiskey victor uniform"))
    assert ud.minhash_signature(a) != ud.minhash_signature(b)


# ── lsh_bands ──────────────────────────────────────────────────────────────────


def test_lsh_bands_has_length_b():
    sig = ud.minhash_signature(ud.shingles(ud.normalize_text("one two three four five")))
    assert len(ud.lsh_bands(sig, b=8)) == 8
    assert len(ud.lsh_bands(sig, b=4)) == 4


def test_lsh_bands_is_deterministic():
    sig = ud.minhash_signature(ud.shingles(ud.normalize_text("one two three four five")))
    assert ud.lsh_bands(sig) == ud.lsh_bands(sig)


def test_lsh_bands_identical_signatures_produce_identical_bands():
    a = ud.minhash_signature(ud.shingles(ud.normalize_text("Hello   World Foo Bar Baz")))
    b = ud.minhash_signature(ud.shingles(ud.normalize_text("hello world\nfoo bar baz")))
    assert ud.lsh_bands(a) == ud.lsh_bands(b)


def test_lsh_bands_differ_for_very_different_documents():
    sig_a = ud.minhash_signature(
        ud.shingles(ud.normalize_text("alpha beta gamma delta epsilon zeta eta theta"))
    )
    sig_b = ud.minhash_signature(
        ud.shingles(ud.normalize_text("zulu yankee xray whiskey victor uniform tango sierra"))
    )
    assert ud.lsh_bands(sig_a) != ud.lsh_bands(sig_b)


def test_lsh_bands_of_empty_signature_returns_b_empty_strings():
    assert ud.lsh_bands([], b=8) == [""] * 8


def test_lsh_bands_near_duplicate_documents_collide_on_at_least_one_band():
    """The whole point of LSH banding — a document edited by adding one
    trailing sentence should still collide on at least one band with its
    near-identical sibling, so `find_update_shortlist`'s OR-across-bands
    lookup can find it as a candidate."""
    base = " ".join(f"word{i}" for i in range(100))
    sig_a = ud.minhash_signature(ud.shingles(ud.normalize_text(base)))
    sig_b = ud.minhash_signature(
        ud.shingles(ud.normalize_text(base + " one small addition at the end"))
    )
    bands_a = set(ud.lsh_bands(sig_a, b=8))
    bands_b = set(ud.lsh_bands(sig_b, b=8))
    assert bands_a & bands_b  # at least one shared band fingerprint
