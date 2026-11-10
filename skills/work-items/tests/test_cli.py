"""CLI wiring: resolve / move / promote / flush / init (subprocess, real entry point)."""
import json
import subprocess
import sys

import pytest

from conftest import SCRIPTS
from wilib import events
from wilib.store import Store

WI = SCRIPTS / "wi.py"
KEYS = {"id", "uid", "type", "parent", "aliases", "slug", "home", "touched",
        "milestone", "where", "links", "retired", "matched_as", "input"}


def wi(cwd, *args):
    return subprocess.run([sys.executable, str(WI), *args], cwd=cwd, capture_output=True, text=True)


@pytest.fixture
def proj(repo):
    assert wi(repo, "init", "--key", "GAL", "--components", "a,b").returncode == 0
    for a in (("new", "I"), ("new", "I"), ("new", "F", "--parent", "GAL-I03", "--slug", "feat"),
              ("new", "S", "--parent", "GAL-I03-F0001")):
        assert wi(repo, *a).returncode == 0
    return repo


def test_init_creates_project_without_key_claim(repo):
    r = wi(repo, "init", "--key", "GAL", "--components", "a,b")
    assert r.returncode == 0, r.stderr
    pj = json.loads((repo / ".work-items" / "project.json").read_text())
    assert pj["key"] == "GAL" and pj["components"] == ["a", "b"]
    assert wi(repo, "new", "I").stdout == "GAL-I03\n"


def test_init_accepts_space_separated_components_and_rejects_bad_key(repo):
    assert wi(repo, "init", "--key", "bad").returncode == 1
    assert wi(repo, "init", "--key", "GAL", "--components", "x", "y").returncode == 0
    assert json.loads((repo / ".work-items" / "project.json").read_text())["components"] == ["x", "y"]


def test_init_refuses_other_key(repo):
    wi(repo, "init", "--key", "GAL")
    r = wi(repo, "init", "--key", "ZZZ")
    assert r.returncode == 1 and "GAL" in r.stderr


def test_flush_writes_tracked_registry(proj):
    (proj / ".work-items" / "registry.jsonl").unlink()
    r = wi(proj, "flush")
    assert r.returncode == 0, r.stderr
    ids = sorted(e["id"] for e in events.read_events(
        (proj / ".work-items" / "registry.jsonl").read_text()) if e["ev"] == "new")
    assert ids == ["GAL-I01", "GAL-I02", "GAL-I03", "GAL-I03-F0001",
                   "GAL-I03-F0001-S01", "GAL-I04"]


def test_move_prints_new_id_and_resolve_text_says_alias(proj):
    assert wi(proj, "move", "GAL-I03-F0001", "GAL-I04").stdout == "GAL-I04-F0001\n"
    out = wi(proj, "resolve", "GAL-I03-F0001-S01").stdout
    assert out.startswith("GAL-I04-F0001-S01") and "alias of GAL-I04-F0001-S01" in out
    assert wi(proj, "resolve", "GAL-I04-F0001").stdout == "GAL-I04-F0001\n"


def test_resolve_json_is_exactly_the_contract(proj):
    wi(proj, "move", "GAL-I03-F0001", "none")
    d = json.loads(wi(proj, "resolve", "GAL-I03-F0001", "--json").stdout)
    assert set(d) == KEYS
    assert (d["id"], d["matched_as"], d["input"], d["aliases"]) == (
        "GAL-F0001", "alias", "GAL-I03-F0001", ["GAL-I03-F0001"])


def test_resolve_unknown_exit_4_and_ambiguous_exit_3(proj):
    r = wi(proj, "resolve", "GAL-F0999")
    assert r.returncode == 4 and "unknown" in r.stderr
    # a duplicate slug can only arrive through a merge now: append the event directly
    with Store(proj, "GAL").transaction() as tx:
        tx.append({"ev": "new", "id": "GAL-F0002", "type": "F", "parent": None,
                   "home": "repo", "slug": "feat"})
    assert wi(proj, "resolve", "feat").returncode == 3


def test_move_illegal_parent_exit_1_and_promote(proj):
    r = wi(proj, "move", "GAL-I03-F0001", "GAL-I01")
    assert r.returncode == 1 and "wi:" in r.stderr
    assert wi(proj, "new", "K").stdout == "GAL-I02-K0001\n"
    assert wi(proj, "promote", "GAL-I02-K0001", "--to", "F", "--parent", "GAL-I04").stdout \
        == "GAL-I04-F0002\n"
