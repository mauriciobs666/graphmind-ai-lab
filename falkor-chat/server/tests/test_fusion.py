"""Unit tests for `fusion.py` (K-050 M5 Stage 4, FR-9).

`find_fuzzy_candidates` holds no Cypher itself — it builds the app-side fuzzy
query string and delegates to `repo.find_fuzzy_candidates` (the real Cypher is
proven live in `test_repository.py`). Here the focus is the query-string
construction and `classify_fuzzy`'s degenerate-input handling — both pure,
fake-repo-testable logic.

The K-068 section at the bottom is the exception: whether RediSearch *accepts*
the built query, and whether it still finds the indexed name, is a property of
the real full-text index a fake repo cannot prove — so those cases run live
against `ws:test` through the `repo` fixture (FalkorDB required; `ws:test` only,
never the global `reference` graph).
"""

from __future__ import annotations

import re
import unicodedata

import pytest

from falkorchat import fusion


class _RecordingRepo:
    def __init__(self, result=None):
        self.calls: list[tuple] = []
        self._result = result if result is not None else []

    def find_fuzzy_candidates(self, ws, *, fuzzy_query, type, limit=5):
        self.calls.append((ws, fuzzy_query, type, limit))
        return self._result


# ── find_fuzzy_candidates: query-string construction ─────────────────────────


def test_find_fuzzy_candidates_wraps_a_single_token_name():
    repo = _RecordingRepo()

    fusion.find_fuzzy_candidates(repo, "acme", "Acme", "Organization")

    assert repo.calls == [("acme", "%Acme%", "Organization", 5)]


def test_find_fuzzy_candidates_wraps_each_token_of_a_multi_word_name():
    repo = _RecordingRepo()

    fusion.find_fuzzy_candidates(repo, "acme", "Acme Corp International", "Organization")

    assert repo.calls[0][1] == "%Acme% %Corp% %International%"


def test_find_fuzzy_candidates_passes_through_type_and_limit():
    repo = _RecordingRepo()

    fusion.find_fuzzy_candidates(repo, "acme", "Bob", "Person", limit=10)

    ws, query, type_, limit = repo.calls[0]
    assert type_ == "Person"
    assert limit == 10


def test_find_fuzzy_candidates_returns_the_repository_result_unmodified():
    candidates = [{"entityId": "e1", "name": "Acme", "type": "Organization", "score": 2.0}]
    repo = _RecordingRepo(result=candidates)

    result = fusion.find_fuzzy_candidates(repo, "acme", "Acme", "Organization")

    assert result == candidates


def test_find_fuzzy_candidates_skips_the_repo_call_for_an_empty_name():
    # A blank/whitespace-only name should never reach extraction in practice
    # (extraction._coerce_entities drops it), but this is the defensive
    # posture documented on `_fuzzy_query` — an empty query string is treated
    # as "no candidates," never issued to RediSearch.
    repo = _RecordingRepo()

    result = fusion.find_fuzzy_candidates(repo, "acme", "   ", "Organization")

    assert result == []
    assert repo.calls == []


# ── classify_fuzzy ────────────────────────────────────────────────────────────


def test_classify_fuzzy_none_when_no_candidates():
    assert fusion.classify_fuzzy([]) == "none"


def test_classify_fuzzy_suggested_when_any_candidate_exists():
    candidates = [{"entityId": "e1", "name": "Acme", "type": "Organization", "score": 0.1}]
    assert fusion.classify_fuzzy(candidates) == "suggested"


# ── K-068: RediSearch query syntax in entity names ────────────────────────────
#
# Every name below is a real extracted entity name whose suggested-tier lookup
# raised `RediSearch: Syntax error` in the 2026-10-17 K-071 re-drive of
# `ws:agent-team` (one per trigger class), plus one synthetic name carrying the
# remaining metacharacters a read-only probe of the live parser showed it
# rejects (`| ~ @ $ * ? \\ ` & # ^ ! + > , ;`) that no real name happened to hit.

K068_REAL_FAILING_NAMES = [
    ("model-bench", "Concept"),  # hyphen between words
    ("K-067", "Concept"),  # hyphen before digits (`near -067`)
    ("2026-10-15", "Event"),  # date
    ("Message.toolsUsed", "Concept"),  # dot
    ("exec.py", "Product"),  # dot before an extension
    ("retry.is_transient", "Concept"),  # dot + underscore (underscore is a word char)
    ("qwen/qwen3-4b-5007", "Product"),  # slash
    ("HTTP 400/500", "Concept"),  # slash between digits
    ("2026-10-15T09:27:50-03:00", "Event"),  # colon
    ("Unloading model: <id>", "Other"),  # colon + angle brackets
    ("--name", "Concept"),  # leading punctuation (`offset 1 near ` with empty tail)
    ('[INFO] Returning {"data": [...]', "Other"),  # brackets, braces, quotes
    ("uvicorn's LOGGING_CONFIG", "Other"),  # apostrophe
    ("43%", "Other"),  # percent (collides with the `%term%` fuzzy wrapper)
    ("rc=1", "Other"),  # equals
    ("install()", "Product"),  # parentheses
    ("[LMSAuthenticator][Client=lms-cli]\u2026", "Other"),  # non-ASCII ellipsis
    ("A|B ~C @D $E *F ?G \\H `I &J #K ^L !M +N >O ,P ;Q", "Concept"),  # synthetic
    # Not in the re-drive log, but the only `ws:agent-team` name the K-068 replay
    # of all 7450 entity names still broke after the punctuation split: `>_<`
    # leaves a lone `_` token, and `%_%` is itself a syntax error.
    ("__editable___<pkg>_<ver>_finder.install()", "Other"),
]


def _seed_entity(repo, entity_id: str, name: str, type: str) -> None:
    repo.create_entity(
        "test", entity_id=entity_id, name=name, name_normalized=name.lower(),
        type=type, created_at=100,
    )


@pytest.mark.parametrize(("name", "type_"), K068_REAL_FAILING_NAMES)
def test_find_fuzzy_candidates_live_accepts_a_real_punctuated_name_and_finds_it(
    repo, name, type_,
):
    # An existing entity of exactly this name is indexed; the new entity's
    # fuzzy lookup must both parse (no `RediSearch: Syntax error`) and reach
    # it — query terms aligned with how RediSearch tokenized the indexed name.
    _seed_entity(repo, "existing", name, type_)

    results = fusion.find_fuzzy_candidates(repo, "test", name, type_)

    assert "existing" in [r["entityId"] for r in results]


def test_find_fuzzy_candidates_live_still_matches_a_plain_name_one_edit_away(repo):
    _seed_entity(repo, "existing", "Acme Corporation", "Organization")

    results = fusion.find_fuzzy_candidates(repo, "test", "Acmee Corporation", "Organization")

    assert [r["entityId"] for r in results] == ["existing"]


@pytest.mark.parametrize(
    ("punctuated", "plain"),
    [
        ("model-bench", "model bench"),
        ("lms.exe", "lms exe"),
        ("python-web-quirks", "Python Web Quirks"),
    ],
)
def test_find_fuzzy_candidates_live_punctuated_name_finds_its_plain_word_counterpart(
    repo, punctuated, plain,
):
    _seed_entity(repo, "plain", plain, "Concept")

    results = fusion.find_fuzzy_candidates(repo, "test", punctuated, "Concept")

    assert [r["entityId"] for r in results] == ["plain"]


@pytest.mark.parametrize(
    "name", ["--", "...", "/", "()", " - : ", "_", "<_>", "\\", "\x01\n"],
)
def test_find_fuzzy_candidates_skips_the_repo_call_for_an_all_punctuation_name(name):
    repo = _RecordingRepo()

    result = fusion.find_fuzzy_candidates(repo, "acme", name, "Concept")

    assert result == []
    assert repo.calls == []


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model-bench FX1", "%model% %bench% %FX1%"),
        ("retry.is_transient", "%retry% %is_transient%"),
        ("--name", "%name%"),
        ("43%", "%43%"),
        ("caf\u00e9-bar", "%caf\u00e9% %bar%"),
        ("a <_> b", "%a% %b%"),
        # Non-ASCII punctuation is kept inside a term — the index does too.
        ("A1\u2013A15", "%A1\u2013A15%"),
        ("\u2026", "%\u2026%"),
        # `\` and ASCII control characters (tab aside) are deleted, not split on.
        ("a\\b", "%ab%"),
        ("line\nbreak", "%linebreak%"),
        ("tab\tsep", "%tab% %sep%"),
        ("_private __dunder__", "%_private% %__dunder__%"),
    ],
)
def test_fuzzy_query_tokenizes_like_the_entity_name_index(name, expected):
    assert fusion._fuzzy_query(name) == expected


# ── K-068 pass 2: separator set measured against the live index ───────────────
#
# For each character `c`, an entity named `aa<c>bb` is indexed and its own name is
# looked up: whatever the index does with `c` (separate on it, keep it inside the
# term, drop it), the built query must parse and find the stored name. Samples:
# every ASCII non-alphanumeric character but NUL, plus the first character of
# each Unicode general category with no word characters. The emoji case is an
# accepted limitation: RediSearch misses it under every builder tried.


def _first_non_word_char_per_unicode_category() -> list[str]:
    seen: dict[str, str] = {}
    for cp in range(0x80, 0x30000):
        ch = chr(cp)
        cat = unicodedata.category(ch)
        if cat in ("Cs", "Co", "Cn") or re.match(r"\w", ch):
            continue
        seen.setdefault(cat, ch)
    return list(seen.values())


_CHAR_CLASS_SAMPLES = [
    chr(c) for c in range(1, 128) if not chr(c).isalnum()
] + _first_non_word_char_per_unicode_category()


@pytest.mark.parametrize("c", _CHAR_CLASS_SAMPLES, ids=lambda c: f"U+{ord(c):04X}")
def test_find_fuzzy_candidates_live_finds_a_stored_name_containing_any_character_class(repo, c):
    name = f"aa{c}bb"
    _seed_entity(repo, "existing", name, "Concept")

    results = fusion.find_fuzzy_candidates(repo, "test", name, "Concept")

    assert "existing" in [r["entityId"] for r in results]


@pytest.mark.xfail(
    strict=True,
    reason="accepted limitation: an emoji inside a stored name is missed by every builder tried",
)
def test_find_fuzzy_candidates_live_emoji_inside_a_name_is_a_known_miss(repo):
    _seed_entity(repo, "existing", "aa\U0001F600bb", "Concept")

    results = fusion.find_fuzzy_candidates(repo, "test", "aa\U0001F600bb", "Concept")

    assert "existing" in [r["entityId"] for r in results]


# Real `ws:agent-team` names that parsed and found themselves before K-068 and
# stopped doing so under a split on every non-word character: the index keeps
# non-ASCII punctuation inside a term.
K068_NON_ASCII_REAL_NAMES = [
    "A1\u2013A15",
    "Passes 5\u20139",
    "producer\u2192store\u2192load\u2192consumer round trip",
    "A\u2229B",
    "\u2205",
]


@pytest.mark.parametrize("name", K068_NON_ASCII_REAL_NAMES)
def test_find_fuzzy_candidates_live_non_ascii_punctuated_real_name_finds_itself(repo, name):
    _seed_entity(repo, "existing", name, "Concept")

    results = fusion.find_fuzzy_candidates(repo, "test", name, "Concept")

    assert "existing" in [r["entityId"] for r in results]


def test_find_fuzzy_candidates_live_does_not_return_an_entity_sharing_only_one_token(repo):
    # Precision: the per-token terms are ANDed (K-064 owns any change to that),
    # so a same-type entity sharing just one of the punctuated name's words is
    # not a candidate.
    _seed_entity(repo, "target", "model-bench", "Concept")
    _seed_entity(repo, "decoy-a", "bench press", "Concept")
    _seed_entity(repo, "decoy-b", "model railway", "Concept")

    results = fusion.find_fuzzy_candidates(repo, "test", "model-bench", "Concept")

    assert [r["entityId"] for r in results] == ["target"]
