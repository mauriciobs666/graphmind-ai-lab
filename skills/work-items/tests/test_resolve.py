"""resolve: every name kind, FR/AC suffix split, the pinned --json contract (plan 3.9, step 4)."""
import json

import pytest

from wilib import alloc, project, resolve
from wilib.store import Store

KEYS = {"id", "uid", "type", "parent", "aliases", "slug", "home", "touched",
        "milestone", "where", "links", "retired", "matched_as", "input"}


@pytest.fixture
def proj(repo):
    project.init(repo, "GAL", ["a"])
    alloc.new(repo, "I")                              # GAL-I03
    alloc.new(repo, "I")                              # GAL-I04
    alloc.new(repo, "F", parent="GAL-I03", slug="feat")   # GAL-I03-F0001
    return repo


def st(root):
    return Store(root, "GAL").read_state()


def test_current_id(proj):
    r = resolve.resolve(st(proj), "GAL-I03-F0001")
    assert (r["id"], r["uid"], r["type"], r["parent"], r["matched_as"]) == (
        "GAL-I03-F0001", "GAL-I03-F0001", "F", "GAL-I03", "current")
    assert r["slug"] == "feat" and r["input"] == "GAL-I03-F0001" and r["aliases"] == []


def test_alias_reports_current(proj):
    from wilib import move
    move.move(proj, "GAL-I03-F0001", "GAL-I04")
    r = resolve.resolve(st(proj), "GAL-I03-F0001")
    assert (r["id"], r["matched_as"], r["input"], r["uid"]) == (
        "GAL-I04-F0001", "alias", "GAL-I03-F0001", "GAL-I03-F0001")
    assert r["aliases"] == ["GAL-I03-F0001"]


def test_slug(proj):
    r = resolve.resolve(st(proj), "feat")
    assert (r["id"], r["matched_as"]) == ("GAL-I03-F0001", "slug")


def test_unknown_raises_code_4(proj):
    with pytest.raises(resolve.ResolveError) as e:
        resolve.resolve(st(proj), "GAL-F9999")
    assert e.value.code == 4


def test_duplicate_slug_is_ambiguous_code_3(proj):
    alloc.new(proj, "F", slug="feat")
    with pytest.raises(resolve.ResolveError) as e:
        resolve.resolve(st(proj), "feat")
    assert e.value.code == 3


def test_requirement_citation_current(proj):
    r = resolve.resolve(st(proj), "GAL-I03-F0001-FR2a")
    assert (r["id"], r["type"], r["matched_as"], r["parent"]) == (
        "GAL-I03-F0001-FR2a", "FR", "current", "GAL-I03-F0001")


def test_requirement_of_unknown_feature_is_unknown(proj):
    with pytest.raises(resolve.ResolveError):
        resolve.resolve(st(proj), "GAL-I03-F0099-AC1")


def test_fallback_hook_is_consulted_last(proj, monkeypatch):
    monkeypatch.setattr(resolve, "FALLBACKS",
                        [lambda s, n: ("GAL-I03-F0001", "legacy") if n == "x:K-1" else None])
    r = resolve.resolve(st(proj), "x:K-1")
    assert (r["id"], r["matched_as"], r["input"]) == ("GAL-I03-F0001", "legacy", "x:K-1")


def test_record_key_set_is_exact_for_plain_item(proj):
    assert set(resolve.resolve(st(proj), "GAL-I03-F0001")) == KEYS


def test_record_key_set_is_exact_with_attributes(proj):
    with Store(proj, "GAL").transaction() as tx:
        tx.append({"ev": "set", "uid": "GAL-I03-F0001", "attr": "milestone", "value": "M3"})
        tx.append({"ev": "set", "uid": "GAL-I03-F0001", "attr": "touched", "value": ["a", "b"]})
        tx.append({"ev": "set", "uid": "GAL-I03-F0001", "attr": "where", "value": "docs/x.md"})
        tx.append({"ev": "link", "rel": "relates", "from": "GAL-I03-F0001", "to": "GAL-I04"})
    r = resolve.resolve(st(proj), "GAL-I03-F0001")
    assert set(r) == KEYS
    assert (r["milestone"], r["touched"], r["where"], r["links"]) == (
        "M3", ["a", "b"], "docs/x.md", {"relates": "GAL-I04"})
    assert set(resolve.resolve(st(proj), "GAL-I03-F0001-AC1")) == KEYS
    assert set(resolve.JSON_KEYS) == KEYS


def test_describe_text(proj):
    from wilib import move
    move.move(proj, "GAL-I03-F0001", "GAL-I04")
    s = st(proj)
    assert resolve.describe(resolve.resolve(s, "GAL-I04-F0001")) == "GAL-I04-F0001"
    assert "alias of GAL-I04-F0001" in resolve.describe(resolve.resolve(s, "GAL-I03-F0001"))
    assert json.dumps(resolve.resolve(s, "feat"))   # JSON-serialisable
