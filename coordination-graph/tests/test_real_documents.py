"""The parsers over every real mirrored document: nothing raises, the shapes hold.

Real repo files are the fixtures here (offline, read-only). The numbers asserted are
floors/shape facts, not counts that move whenever a document is added.
"""
from pathlib import Path

import pytest

from coordgraph.discover import discover
from coordgraph.parse.kb import parse_kb
from coordgraph.parse.ledger import UNKNOWN, parse_ledger
from coordgraph.parse.requirements import parse_requirements
from coordgraph.parse.review import parse_review

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def found():
    return discover(REPO)


def test_discovery_finds_every_kind_in_the_real_repo(found):
    kinds = {f.kind for f in found}
    assert kinds == {"ledger", "review", "requirement", "backlog"}
    assert any(f.path == "docs/BACKLOG.md" for f in found)
    assert sum(f.path.startswith("claude/") and f.path.endswith("/kaizen/plan.md") for f in found) >= 10
    assert not any("/archive/" in f.path or f.path.startswith("deprecated/") for f in found)


def test_every_ledger_parses_and_its_units_are_consistent(found):
    reviews = {f.path for f in found if f.kind == "review"}
    ledgers = [f for f in found if f.kind == "ledger"]
    assert len(ledgers) >= 50
    total = 0
    for f in ledgers:
        d = parse_ledger(f.path, (REPO / f.path).read_bytes(), reviews)
        total += len(d.units)
        keys = [u.key for u in d.units]
        assert len(keys) == len(set(keys)), f.path  # `~n` makes every key unique within a file
        assert set(d.resolved_reviews) <= set(d.cited_reviews), f.path
        assert set(d.cited_reviews) <= reviews, f.path
        for u in d.units:
            assert u.status_token  # always present
        for props in d.unit_props():
            assert len(props["key"].encode()) <= 512
    assert total > 800


def test_every_review_and_requirement_parses(found):
    reviews = [f for f in found if f.kind == "review"]
    verdicts = {"needs-changes": 0, "approve": 0, "approve-with-suggestions": 0, UNKNOWN: 0}
    for f in reviews:
        d = parse_review(f.path, (REPO / f.path).read_bytes())
        verdicts[d.verdict_class] = verdicts.get(d.verdict_class, 0) + 1
        assert d.latest_pass >= 1
    assert verdicts["approve-with-suggestions"] > 50 and verdicts["needs-changes"] > 5
    assert verdicts["approve"] > 20
    reqs = [f for f in found if f.kind == "requirement"]
    statuses = {parse_requirements(f.path, (REPO / f.path).read_bytes()).header.doc_status for f in reqs
                if (REPO / f.path).read_bytes().startswith(b"#")}
    assert {"Ready for design", "archived"} <= statuses


def test_the_coordination_graph_review_ends_on_a_needs_changes_or_later_verdict():
    path = "claude/docs/reviews/coordination-graph.md"
    d = parse_review(path, (REPO / path).read_bytes())
    assert d.latest_pass >= 5 and d.verdict_class != UNKNOWN


def test_every_kb_file_in_the_claude_tree_parses():
    files = sorted(REPO.glob("claude/*/*.md"))
    parsed = [parse_kb(p.relative_to(REPO).as_posix(), p.read_bytes()) for p in files]
    sections = [s for d in parsed for s in d.sections]
    assert len(sections) > 100
    for s in sections:
        assert s.section_hash and (s.mode == "bullets" or s.claims or s.body == "")
