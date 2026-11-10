"""Step 6b: ``headers sync`` (family glob, header-only edits), ``--sync`` as a second phase,
``--home`` validation.  Documents live only under tmp_path git repos."""
import builtins
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import SCRIPTS
from wilib import alloc, cli, headers, project
from wilib.store import Store

WI = SCRIPTS / "wi.py"
HEADER = "> **Status:** active · **Owner:** `tico` · **Tracks:** —"


def wi(cwd, *args):
    return subprocess.run([sys.executable, str(WI), *args], cwd=cwd, capture_output=True, text=True)


@pytest.fixture
def proj(repo):
    project.init(repo, "GAL", ["a", "b"])
    return repo


def doc(root, rel, header=HEADER, body="first body line\n\nsecond body line\n"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"# Title\n\n{header}\n\n{body}", encoding="utf-8")
    return p


def header_line(path):
    return path.read_text(encoding="utf-8").splitlines()[2]


def rest(path):
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    return hashlib.sha256("".join(lines[:2] + lines[3:]).encode()).hexdigest()


# ------------------------------------------------------------------ AC-14
def test_ac14_slug_and_id_resolve_both_ways_and_every_family_header_carries_the_item(proj):
    assert wi(proj, "new", "F", "--slug", "feat").stdout == "GAL-F0001\n"
    docs = [doc(proj, "docs/requirements/feat.md"), doc(proj, "docs/plans/feat.md"),
            doc(proj, "docs/plans/feat-coordination.md"), doc(proj, "docs/test-plans/feat.md"),
            doc(proj, "a/docs/reviews/feat.md"), doc(proj, "b/docs/manuals/feat.md")]
    other = doc(proj, "docs/plans/other.md")
    assert wi(proj, "headers", "sync", "GAL-F0001").returncode == 0
    for d in docs:
        assert header_line(d) == HEADER + " · **Item:** `GAL-F0001`"
    assert header_line(other) == HEADER
    assert wi(proj, "resolve", "feat").stdout == "GAL-F0001  [slug feat]\n"
    assert wi(proj, "resolve", "GAL-F0001").stdout == "GAL-F0001\n"


def test_sync_without_id_covers_every_slugged_item(proj):
    alloc.new(proj, "F", slug="one")
    alloc.new(proj, "F", slug="two")
    alloc.new(proj, "F")                               # no slug: nothing to match
    d1, d2 = doc(proj, "docs/plans/one.md"), doc(proj, "docs/plans/two.md")
    assert wi(proj, "headers", "sync").returncode == 0
    assert header_line(d1).endswith("**Item:** `GAL-F0001`")
    assert header_line(d2).endswith("**Item:** `GAL-F0002`")


# ------------------------------------------------------------------ moves
def test_header_updated_after_move_lists_old_id_as_alias_and_back(proj):
    alloc.new(proj, "I")                               # GAL-I03
    alloc.new(proj, "F", slug="feat")                  # GAL-F0001
    d = doc(proj, "docs/plans/feat.md")
    assert wi(proj, "headers", "sync", "GAL-F0001").returncode == 0
    assert wi(proj, "move", "GAL-F0001", "GAL-I03", "--sync").stdout == "GAL-I03-F0001\n"
    assert header_line(d) == HEADER + " · **Item:** `GAL-I03-F0001` · **Aliases:** `GAL-F0001`"
    assert wi(proj, "move", "GAL-I03-F0001", "none", "--sync").returncode == 0
    assert header_line(d) == HEADER + " · **Item:** `GAL-F0001` · **Aliases:** `GAL-I03-F0001`"


def test_promote_sync_writes_headers_for_the_promoted_item(proj):
    alloc.new(proj, "K", slug="idea")                  # GAL-I02-K0001
    d = doc(proj, "docs/requirements/idea.md")
    assert wi(proj, "promote", "GAL-I02-K0001", "--to", "F", "--sync").stdout == "GAL-F0001\n"
    assert header_line(d) == HEADER + " · **Item:** `GAL-F0001` · **Aliases:** `GAL-I02-K0001`"


def test_promote_and_move_without_sync_touch_no_document(proj):
    alloc.new(proj, "K", slug="idea")
    d = doc(proj, "docs/requirements/idea.md")
    wi(proj, "promote", "GAL-I02-K0001", "--to", "F")
    assert header_line(d) == HEADER


def test_legacy_name_is_listed_among_aliases(proj):
    cfg = proj / ".work-items" / "project.json"
    data = json.loads(cfg.read_text())
    data["legacy_namespaces"] = {"K-": ["a"]}
    cfg.write_text(json.dumps(data))
    alloc.new(proj, "F", slug="old")
    assert wi(proj, "mint-legacy", "a:K-004", "--type", "K").stdout == "GAL-I02-K0001\n"
    assert wi(proj, "set", "GAL-I02-K0001", "slug=legacy-one").returncode == 0
    d = doc(proj, "docs/plans/legacy-one.md")
    assert wi(proj, "headers", "sync", "GAL-I02-K0001").returncode == 0
    assert header_line(d) == HEADER + " · **Item:** `GAL-I02-K0001` · **Aliases:** `a:K-004`"


# ------------------------------------------------------------------ header-only edits
def test_body_is_byte_identical_and_sync_is_idempotent(proj):
    alloc.new(proj, "F", slug="feat")
    body = "Item: not a header\n**Item:** `GAL-F9999` in prose\n\ttabs and trailing  \nend"
    d = doc(proj, "docs/plans/feat.md", body=body)
    before = rest(d)
    headers.sync(proj)
    after_first = d.read_bytes()
    assert rest(d) == before
    assert header_line(d).endswith("**Item:** `GAL-F0001`")
    assert headers.sync(proj) == []                    # nothing changed: no rewrite reported
    assert d.read_bytes() == after_first


def test_existing_item_and_aliases_fields_are_replaced_in_place_not_duplicated(proj):
    alloc.new(proj, "F", slug="feat")
    d = doc(proj, "docs/plans/feat.md",
            header="> **Status:** active · **Item:** `GAL-F0099` · **Owner:** `tico` · "
                   "**Aliases:** `zzz` · **Tracks:** —")
    headers.sync(proj)
    assert header_line(d) == ("> **Status:** active · **Owner:** `tico` · **Tracks:** — · "
                              "**Item:** `GAL-F0001`")


def test_crlf_documents_keep_their_line_endings(proj):
    alloc.new(proj, "F", slug="feat")
    d = proj / "docs" / "plans" / "feat.md"
    d.parent.mkdir(parents=True)
    d.write_bytes(f"# T\r\n\r\n{HEADER}\r\n\r\nbody\r\n".encode())
    headers.sync(proj)
    assert d.read_bytes() == f"# T\r\n\r\n{HEADER} · **Item:** `GAL-F0001`\r\n\r\nbody\r\n".encode()


def test_document_without_a_header_block_is_left_untouched(proj):
    alloc.new(proj, "F", slug="feat")
    d = proj / "docs" / "plans" / "feat.md"
    d.parent.mkdir(parents=True)
    d.write_text("# T\n\njust prose\n")
    assert headers.sync(proj) == []
    assert d.read_text() == "# T\n\njust prose\n"


# ------------------------------------------------------------------ family glob
@pytest.mark.parametrize("order", [("x", "x2"), ("x2", "x")])
def test_family_membership_by_slug_and_longer_registered_slug_exclusion(proj, order):
    for slug in order:                                 # either registration order: no overwrite race
        alloc.new(proj, "F", slug=slug)
    ids = {slug: f"`GAL-F000{n}`" for n, slug in enumerate(order, 1)}
    names = {n: doc(proj, f"docs/plans/{n}.md") for n in
             ("x", "x2", "x2-coordination", "x-coordination", "x3", "xy", "x2-p1")}
    headers.sync(proj)
    got = {n: (header_line(p).rpartition("**Item:** ")[2] if "Item" in header_line(p) else None)
           for n, p in names.items()}
    assert got == {"x": ids["x"], "x2": ids["x2"], "x2-coordination": ids["x2"],
                   "x-coordination": ids["x"], "x3": ids["x"], "xy": None, "x2-p1": None}


def test_only_family_kinds_and_depths_are_searched(proj):
    alloc.new(proj, "F", slug="feat")
    skipped = [doc(proj, "docs/notes/feat.md"), doc(proj, "a/b/docs/plans/feat.md"),
               doc(proj, "docs/plans/sub/feat.md"), doc(proj, "feat.md")]
    headers.sync(proj)
    assert all("Item" not in header_line(p) for p in skipped)


# ------------------------------------------------------------------ plan test 7 / AC-5
@pytest.fixture
def opened(monkeypatch):
    seen = []
    real_io, real_bi, real_os = io.open, builtins.open, os.open

    def rec_io(f, *a, **k):
        seen.append(os.fspath(f)) if isinstance(f, (str, os.PathLike)) else None
        return real_io(f, *a, **k)

    def rec_os(p, *a, **k):
        seen.append(os.fspath(p))
        return real_os(p, *a, **k)

    monkeypatch.setattr(io, "open", rec_io)
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (seen.append(os.fspath(f)) if isinstance(
        f, (str, os.PathLike)) else None, real_bi(f, *a, **k))[1])
    monkeypatch.setattr(os, "open", rec_os)
    return seen


def outside_registry(seen, root):
    return sorted({p for p in map(os.path.abspath, seen)
                   if p.startswith(str(root)) and "/.work-items/" not in p and "/.git/" not in p})


def test_plain_new_and_new_with_slug_read_no_document(proj, opened):
    doc(proj, "docs/plans/feat.md")
    opened.clear()
    assert cli.main(["new", "F"], cwd=proj) == 0
    assert cli.main(["new", "F", "--slug", "feat"], cwd=proj) == 0
    assert opened and outside_registry(opened, proj) == []     # recorder is live, saw only the registry


def test_sync_phase_opens_only_family_pattern_files(proj, opened):
    doc(proj, "docs/plans/feat.md")
    doc(proj, "a/docs/reviews/feat-impl.md")
    doc(proj, "docs/plans/unrelated.md")
    doc(proj, "docs/plans/feat.txt")
    opened.clear()
    assert cli.main(["new", "F", "--slug", "feat", "--sync"], cwd=proj) == 0
    assert outside_registry(opened, proj) == sorted(
        str(proj / p) for p in ("docs/plans/feat.md", "a/docs/reviews/feat-impl.md"))


def test_header_writing_happens_after_the_allocation_lock_is_released(proj, monkeypatch):
    state = {}
    real = Store.transaction

    class Probe:
        def __init__(self, cm):
            self.cm = cm

        def __enter__(self):
            state["in_tx"] = True
            return self.cm.__enter__()

        def __exit__(self, *a):
            state["in_tx"] = False
            return self.cm.__exit__(*a)

    monkeypatch.setattr(Store, "transaction", lambda self: Probe(real(self)))

    def fake_sync(root, id_=None, **k):
        state["sync_in_tx"] = state["in_tx"]
        return []

    monkeypatch.setattr(headers, "sync", fake_sync)
    cli.main(["new", "F", "--slug", "feat", "--sync"], cwd=proj)
    assert state["sync_in_tx"] is False


# ------------------------------------------------------------------ --sync accepted; --home
def test_new_with_sync_allocates_prints_id_and_syncs(proj):
    d = doc(proj, "docs/requirements/feat.md")
    r = wi(proj, "new", "F", "--slug", "feat", "--sync")
    assert (r.returncode, r.stdout) == (0, "GAL-F0001\n")
    assert header_line(d).endswith("**Item:** `GAL-F0001`")


def test_new_home_must_be_a_declared_component_or_repo_and_refusal_burns_no_id(proj):
    r = wi(proj, "new", "F", "--home", "bogus")
    assert r.returncode == 1 and "bogus" in r.stderr
    assert wi(proj, "new", "F", "--home", "a").stdout == "GAL-F0001\n"
    assert wi(proj, "new", "F", "--home", "repo").stdout == "GAL-F0002\n"
    homes = {i.id: i.home for i in Store(proj, "GAL").read_state().items.values()}
    assert (homes["GAL-F0001"], homes["GAL-F0002"]) == ("a", "repo")


def test_init_writes_work_items_gitignore_for_lock_and_tmp(proj):
    assert (proj / ".work-items" / ".gitignore").read_text() == ".lock\n*.tmp\n"
