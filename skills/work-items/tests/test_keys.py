"""AC-10: shared project keys (plan 3.9): suggest/claim refusals, concurrency, init seeds."""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import SCRIPTS

SKILL = SCRIPTS.parent
WI = str(SCRIPTS / "wi.py")


@pytest.fixture
def keys_file(tmp_path):
    p = tmp_path / "keys.json"
    p.write_text("[]\n", encoding="utf-8")
    return p


def wi(keys_file, *args, cwd=None):
    env = {**os.environ, "WI_KEYS_FILE": str(keys_file)}
    return subprocess.run([sys.executable, WI, *args], cwd=cwd, env=env,
                          capture_output=True, text=True)


def listing(keys_file):
    return json.loads(keys_file.read_text(encoding="utf-8"))


def test_claim_records_key_repo_date_and_pool_flag_and_tells_caller_to_commit(keys_file):
    r = wi(keys_file, "key", "claim", "GAL", "--repo", "graphmind-ai-lab")
    assert r.returncode == 0, r.stderr
    (entry,) = listing(keys_file)
    assert entry["key"] == "GAL" and entry["repo"] == "graphmind-ai-lab"
    assert entry["feeds_pool"] is True
    assert len(entry["claimed"]) == 10 and entry["claimed"][4] == "-"   # YYYY-MM-DD
    assert "commit" in r.stdout.lower()


def test_claim_no_pool_flag_records_false(keys_file):
    assert wi(keys_file, "key", "claim", "XYZ", "--repo", "r", "--no-pool").returncode == 0
    assert listing(keys_file)[0]["feeds_pool"] is False


def test_claim_refuses_a_listed_key_and_leaves_file_unchanged(keys_file):
    assert wi(keys_file, "key", "claim", "GAL", "--repo", "a").returncode == 0
    before = keys_file.read_text()
    r = wi(keys_file, "key", "claim", "GAL", "--repo", "b")
    assert r.returncode == 1 and "already" in r.stderr
    assert keys_file.read_text() == before


@pytest.mark.parametrize("key", ["FR", "AC", "TP", "R"])
def test_claim_refuses_reserved_keys(keys_file, key):
    r = wi(keys_file, "key", "claim", key, "--repo", "r")
    assert r.returncode == 1 and "reserved" in r.stderr
    assert listing(keys_file) == []


@pytest.mark.parametrize("key", ["K", "C"])     # the legacy prefixes are one letter: malformed
def test_claim_refuses_one_letter_legacy_prefixes_as_malformed(keys_file, key):
    r = wi(keys_file, "key", "claim", key, "--repo", "r")
    assert r.returncode == 1 and "malformed" in r.stderr
    assert listing(keys_file) == []


@pytest.mark.parametrize("content", [None, "{not json", '{"key": "GAL"}'])
def test_missing_corrupt_or_non_list_keys_file_is_a_clean_error_not_a_traceback(tmp_path, content):
    p = tmp_path / "keys.json"
    if content is not None:
        p.write_text(content)
    for args in (("key", "claim", "GAL", "--repo", "a"), ("key", "suggest", "gal market")):
        r = wi(p, *args)
        assert r.returncode == (0 if content is None and args[1] == "suggest" else 1), (args, r.stderr)
        assert "Traceback" not in r.stderr
        if r.returncode:
            assert r.stderr.startswith("wi: ") and "keys.json" in r.stderr
    assert (not p.exists()) if content is None else p.read_text() == content


@pytest.mark.parametrize("key", ["FR", "AC", "TP", "gal", "G", "TOOLONGX"])
def test_init_refuses_reserved_and_malformed_keys_and_creates_nothing(repo, keys_file, key):
    r = wi(keys_file, "init", "--key", key, cwd=repo)
    assert r.returncode == 1 and r.stderr.startswith("wi: ")
    assert not (repo / ".work-items").exists()


def test_init_accepts_a_key_listed_for_this_repo_without_claiming(repo, keys_file):
    # the tico flow is suggest -> claim -> init, so a listed key must not be refused by init
    assert wi(keys_file, "key", "claim", "GAL", "--repo", "r").returncode == 0
    assert wi(keys_file, "init", "--key", "GAL", cwd=repo).returncode == 0


@pytest.mark.parametrize("key", ["gal", "G", "GALAXYS", "GA1", "G-L", "GAL ", ""])
def test_claim_refuses_malformed_keys(keys_file, key):
    r = wi(keys_file, "key", "claim", key, "--repo", "r")
    assert r.returncode == 1 and "malformed" in r.stderr
    assert listing(keys_file) == []


def test_claim_accepts_boundary_lengths(keys_file):
    assert wi(keys_file, "key", "claim", "AB", "--repo", "r").returncode == 0
    assert wi(keys_file, "key", "claim", "ABCDEF", "--repo", "r").returncode == 0
    assert [e["key"] for e in listing(keys_file)] == ["AB", "ABCDEF"]


def test_claim_writes_only_keys_json(keys_file):
    d = keys_file.parent
    wi(keys_file, "key", "claim", "GAL", "--repo", "r")
    assert sorted(p.name for p in d.iterdir()) == ["keys.json"]


def test_claim_preserves_existing_entries_in_order(keys_file):
    wi(keys_file, "key", "claim", "AAA", "--repo", "one")
    wi(keys_file, "key", "claim", "BBB", "--repo", "two")
    assert [(e["key"], e["repo"]) for e in listing(keys_file)] == [("AAA", "one"), ("BBB", "two")]


def test_suggest_proposes_a_valid_key_and_never_writes(keys_file):
    before = keys_file.read_text()
    r = wi(keys_file, "key", "suggest", "graphmind ai lab")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "GAL"          # initials of the words
    assert keys_file.read_text() == before
    assert sorted(p.name for p in keys_file.parent.iterdir()) == ["keys.json"]


def test_suggest_skips_a_listed_key(keys_file):
    wi(keys_file, "key", "claim", "GAL", "--repo", "r")
    out = wi(keys_file, "key", "suggest", "graphmind ai lab").stdout.strip()
    assert out != "GAL" and out.isalpha() and out.isupper() and 2 <= len(out) <= 6


def test_suggest_skips_reserved_words(keys_file):
    # initials "AC" is reserved; the proposal must not be it
    out = wi(keys_file, "key", "suggest", "acme corp").stdout.strip()
    assert out not in {"AC", "FR", "TP", "R"} and out


def test_suggest_single_word_uses_leading_letters(keys_file):
    assert wi(keys_file, "key", "suggest", "salesperson").stdout.strip() == "SAL"


def test_suggest_with_no_letters_fails(keys_file):
    r = wi(keys_file, "key", "suggest", "123 !!")
    assert r.returncode == 1 and r.stdout == ""


CHILD = """
import sys, time
sys.path.insert(0, sys.argv[1])
from wilib import keys
keys._after_read = lambda: time.sleep(0.4)      # widen the read-check-write window
start = float(sys.argv[3])
time.sleep(max(0, start - time.time()))
try:
    keys.claim("GAL", "repo-" + sys.argv[4], path=sys.argv[2])
    print("won")
except keys.KeyRefused:
    print("lost")
"""


def test_two_concurrent_claims_of_one_key_exactly_one_wins(keys_file):
    start = time.time() + 1.0
    procs = [subprocess.Popen([sys.executable, "-c", CHILD, str(SCRIPTS), str(keys_file),
                               str(start), str(i)], stdout=subprocess.PIPE, text=True)
             for i in range(2)]
    outs = sorted(p.communicate()[0].strip() for p in procs)
    assert outs == ["lost", "won"]
    assert [e["key"] for e in listing(keys_file)] == ["GAL"]


def test_keys_path_defaults_to_the_skill_copy_through_a_symlink(tmp_path):
    link = tmp_path / "skills" / "work-items"
    link.parent.mkdir()
    link.symlink_to(SKILL)
    code = ("import sys; sys.path.insert(0, sys.argv[1]); from wilib import keys; "
            "print(keys.keys_path())")
    env = {k: v for k, v in os.environ.items() if k != "WI_KEYS_FILE"}
    r = subprocess.run([sys.executable, "-c", code, str(link / "scripts")],
                       capture_output=True, text=True, env=env)
    assert Path(r.stdout.strip()) == SKILL / "keys.json"


def test_shipped_keys_json_is_a_valid_key_list():
    entries = json.loads((SKILL / "keys.json").read_text(encoding="utf-8"))
    assert isinstance(entries, list)
    keys = [e["key"] for e in entries]
    assert len(keys) == len(set(keys))                       # a key is claimed once
    for e in entries:
        assert set(e) == {"key", "repo", "claimed", "feeds_pool"}
        assert re.fullmatch(r"[A-Z]{2,6}", e["key"]) and e["key"] not in {"FR", "AC", "TP", "R"}


def test_init_seeds_defects_and_kaizen_and_does_not_touch_the_key_list(repo, keys_file):
    before = keys_file.read_text()
    r = wi(keys_file, "init", "--key", "ZZZ", cwd=repo)
    assert r.returncode == 0, r.stderr
    reg = [json.loads(l) for l in (repo / ".work-items" / "registry.jsonl").read_text().splitlines()]
    assert {(e["id"], e["title"]) for e in reg} == {("ZZZ-I01", "Defects"), ("ZZZ-I02", "Kaizen")}
    assert keys_file.read_text() == before          # init never claims (claim follows agreement)
