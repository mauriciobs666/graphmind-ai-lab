"""Unit tests for `fusion.py` (K-050 M5 Stage 4, FR-9).

`find_fuzzy_candidates` holds no Cypher itself — it builds the app-side fuzzy
query string and delegates to `repo.find_fuzzy_candidates` (the real Cypher is
proven live in `test_repository.py`). Here the focus is the query-string
construction and `classify_fuzzy`'s degenerate-input handling — both pure,
fake-repo-testable logic.
"""

from __future__ import annotations

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
