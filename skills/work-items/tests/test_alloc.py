import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import SCRIPTS
from wilib import alloc, cli, events, project, rules
from wilib.store import Store

WI = SCRIPTS / "wi.py"


@pytest.fixture
def proj(repo):
    project.init(repo, "GAL", ["falkor-chat", "repo"])
    return repo


def wi(cwd, *args):
    r = subprocess.run([sys.executable, str(WI), *args], cwd=cwd, capture_output=True, text=True)
    return r


def state(root):
    return Store(root, "GAL").read_state()


# ---- AC-1: formats --------------------------------------------------------
def test_ac1_first_user_initiative_is_i03(proj):
    assert alloc.new(proj, "I") == "GAL-I03"


def test_ac1_feature_story_task_chain_under_i03(proj):
    assert alloc.new(proj, "I") == "GAL-I03"
    f = alloc.new(proj, "F", parent="GAL-I03", slug="x")
    assert f == "GAL-I03-F0001"
    s = alloc.new(proj, "S", parent=f)
    assert s == "GAL-I03-F0001-S01"
    assert alloc.new(proj, "T", parent=s) == "GAL-I03-F0001-S01-T01"


def test_ac1_feature_under_no_initiative(proj):
    assert alloc.new(proj, "F", slug="solo") == "GAL-F0001"
    assert alloc.new(proj, "F") == "GAL-F0002"


def test_defect_and_kaizen_land_under_their_initiatives(proj):
    assert alloc.new(proj, "D") == "GAL-I01-D0001"
    assert alloc.new(proj, "K") == "GAL-I02-K0001"
    assert alloc.new(proj, "D") == "GAL-I01-D0002"


def test_task_under_feature_defect_and_kaizen(proj):
    f = alloc.new(proj, "F")
    assert alloc.new(proj, "T", parent=f) == "GAL-F0001-T01"
    d = alloc.new(proj, "D")
    assert alloc.new(proj, "T", parent=d) == "GAL-I01-D0001-T01"
    k = alloc.new(proj, "K")
    assert alloc.new(proj, "T", parent=k) == "GAL-I02-K0001-T01"


def test_feature_counter_is_project_wide_across_initiatives(proj):
    alloc.new(proj, "I")
    alloc.new(proj, "I")
    assert alloc.new(proj, "F", parent="GAL-I03") == "GAL-I03-F0001"
    assert alloc.new(proj, "F", parent="GAL-I04") == "GAL-I04-F0002"


def test_local_counters_are_per_parent(proj):
    f1, f2 = alloc.new(proj, "F"), alloc.new(proj, "F")
    assert alloc.new(proj, "S", parent=f1) == "GAL-F0001-S01"
    assert alloc.new(proj, "S", parent=f1) == "GAL-F0001-S02"
    assert alloc.new(proj, "S", parent=f2) == "GAL-F0002-S01"


def test_slug_title_home_recorded(proj):
    alloc.new(proj, "F", slug="my-slug", title="My title", home="falkor-chat")
    it = state(proj).items["GAL-F0001"]
    assert (it.slug, it.title, it.home) == ("my-slug", "My title", "falkor-chat")


# ---- AC-13 ---------------------------------------------------------------
def test_ac13_hundredth_story_is_s100_and_earlier_ids_unchanged(proj):
    f = alloc.new(proj, "F")
    ids = [alloc.new(proj, "S", parent=f) for _ in range(100)]
    assert ids[0] == "GAL-F0001-S01"
    assert ids[98] == "GAL-F0001-S99"
    assert ids[99] == "GAL-F0001-S100"
    assert len(set(ids)) == 100
    st = state(proj)
    for n in (1, 9, 10, 99):
        assert st.items[f"GAL-F0001-S{n:02d}"].id == f"GAL-F0001-S{n:02d}"


# ---- AC-4 ----------------------------------------------------------------
def test_ac4_retired_number_never_reissued_and_id_resolves(proj):
    f = alloc.new(proj, "F")
    s1 = alloc.new(proj, "S", parent=f)
    s2 = alloc.new(proj, "S", parent=f)
    assert alloc.retire(proj, s2) == "GAL-F0001-S02"
    assert alloc.new(proj, "S", parent=f) == "GAL-F0001-S03"
    st = state(proj)
    assert st.items[st.resolve_uid("GAL-F0001-S02")].retired is True
    assert st.items[st.resolve_uid(s1)].retired is False


def test_ac4_retired_global_number_not_reissued(proj):
    f = alloc.new(proj, "F")
    alloc.retire(proj, f)
    assert alloc.new(proj, "F") == "GAL-F0002"


def test_retire_unknown_refused(proj):
    with pytest.raises(alloc.AllocError):
        alloc.retire(proj, "GAL-F0999")


# ---- AC-5: allocation reads only registry files ---------------------------
def test_ac5_new_reads_nothing_outside_registry_and_journal(proj, monkeypatch):
    docs = proj / "docs" / "requirements"
    docs.mkdir(parents=True)
    (docs / "a.md").write_text("**Item:** `GAL-F0500`\n")
    (proj / "docs" / "BACKLOG.md").write_text("GAL-F9999\n")
    bad = proj / "docs" / "unreadable.md"
    bad.write_text("x")
    bad.chmod(0)
    (proj / "docs" / "reviews").mkdir()
    seen = []
    real_open, real_osopen, real_scandir = io.open, os.open, os.scandir

    def rec_open(f, *a, **k):
        if not isinstance(f, int):      # int = an fd (subprocess pipes), not a path
            seen.append(str(f))
        return real_open(f, *a, **k)

    def rec_osopen(p, *a, **k):
        seen.append(str(p))
        return real_osopen(p, *a, **k)

    def rec_scandir(p="."):
        seen.append(str(p))
        return real_scandir(p)

    monkeypatch.setattr(io, "open", rec_open)
    monkeypatch.setattr("builtins.open", rec_open)
    monkeypatch.setattr(os, "open", rec_osopen)
    monkeypatch.setattr(os, "scandir", rec_scandir)
    monkeypatch.setattr(os, "listdir", lambda *a, **k: seen.append(str(a)) or [])
    assert alloc.new(proj, "F") == "GAL-F0001"
    assert alloc.new(proj, "F", slug="s") == "GAL-F0002"
    bad.chmod(0o644)
    assert seen, "recorder saw nothing: test is vacuous"
    git_common = (proj / ".git").resolve()
    allowed = (str(proj / ".work-items"), str(git_common))
    outside = [p for p in seen if not os.path.abspath(p).startswith(allowed)]
    assert outside == []


# ---- AC-16 via the CLI -----------------------------------------------------
def test_cli_new_prints_id_with_parent_argument(proj):
    assert wi(proj, "new", "I").stdout == "GAL-I03\n"
    r = wi(proj, "new", "F", "--parent", "GAL-I03", "--slug", "s")
    assert (r.returncode, r.stdout) == (0, "GAL-I03-F0001\n")
    r = wi(proj, "new", "S", "--parent", "GAL-I03-F0001")
    assert r.stdout == "GAL-I03-F0001-S01\n"
    assert wi(proj / ".work-items", "new", "T", "--parent", "GAL-I03-F0001-S01").stdout == \
        "GAL-I03-F0001-S01-T01\n"


@pytest.mark.parametrize("args", [
    ("F", "--parent", "GAL-I01"),
    ("F", "--parent", "GAL-I02"),
    ("S", "--parent", "GAL-I01-D0001"),
    ("S", "--parent", "GAL-I02-K0001"),
    ("S",),
    ("T",),
    ("I", "--parent", "GAL-F0001"),
    ("D", "--parent", "GAL-F0001"),
])
def test_ac16_cli_refuses_illegal_parent(proj, args):
    alloc.new(proj, "F")                    # GAL-F0001
    alloc.new(proj, "D")                    # GAL-I01-D0001
    alloc.new(proj, "K")                    # GAL-I02-K0001
    before = len(state(proj).items)
    r = wi(proj, "new", *args)
    assert r.returncode != 0
    assert r.stderr.startswith("wi: ") and "parent" in r.stderr
    assert len(state(proj).items) == before


def test_cli_unknown_parent_refused(proj):
    r = wi(proj, "new", "S", "--parent", "GAL-F0042")
    assert r.returncode != 0 and "unknown parent" in r.stderr


def test_cli_retire_and_no_reissue(proj):
    wi(proj, "new", "F")
    assert wi(proj, "new", "S", "--parent", "GAL-F0001").stdout == "GAL-F0001-S01\n"
    assert wi(proj, "retire", "GAL-F0001-S01").returncode == 0
    assert wi(proj, "new", "S", "--parent", "GAL-F0001").stdout == "GAL-F0001-S02\n"


def test_sync_is_accepted_and_the_allocated_id_is_still_printed(proj):
    r = wi(proj, "new", "F", "--slug", "s", "--sync")
    assert (r.returncode, r.stdout) == (0, "GAL-F0001\n")


def test_illegal_parent_message_uses_padded_ids(proj):
    alloc.new(proj, "I")
    with pytest.raises(rules.IllegalParent, match=r"T may not have I03 as parent"):
        alloc.new(proj, "T", parent="GAL-I03")
