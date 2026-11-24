"""D9b — owner protection of the installed manifest (egress-spec5 section 1, section 7 item 8): real files, directories,
symlinks and FIFOs in a temporary directory with an injected trusted uid (the real root-owned `/etc/llmgateway` is tier 2).
`ccg.broker.ownerprot` is the broker's own implementation (it never imports `llmgateway`); the parity test feeds it and
`llmgateway.ownercheck.read_owner_file` the same planted files. An unimportable `llmgateway` FAILS that test."""
from __future__ import annotations

import os

import pytest

from ccg.broker import ownerprot
from ccg.broker.audit import PreconditionError
from ccg.broker.ownerprot import read_owner_file, require_owner_file

ME = os.getuid()
TRUSTED = frozenset({ME})
OTHER = frozenset({ME + 4242})


@pytest.fixture
def root(tmp_path):
    """A trusted chain `<tmp>/root/etc/system.yaml`, all 0755/0644, owned by this uid."""
    r = tmp_path / "root"
    (r / "etc").mkdir(parents=True)
    r.chmod(0o755)
    (r / "etc").chmod(0o755)
    f = r / "etc" / "system.yaml"
    f.write_text("a: 1\n")
    f.chmod(0o644)
    return r


def check(root, path=None, trusted=TRUSTED, **kw):
    return read_owner_file(path or root / "etc" / "system.yaml", trusted, anchor=root, **kw)


def test_a_protected_file_is_read_from_its_descriptor(root):
    res = check(root)
    assert res.ok and res.data == b"a: 1\n" and res.reason == ""


@pytest.mark.parametrize("mode", [0o664, 0o646, 0o666, 0o620, 0o602])
def test_a_group_or_other_writable_file_is_refused_with_no_bytes(root, mode):
    (root / "etc" / "system.yaml").chmod(mode)
    res = check(root)
    assert not res.ok and res.reason == "file is group- or other-writable" and res.data == b""


def test_a_file_of_an_untrusted_owner_is_refused(root):
    res = check(root, trusted=OTHER)
    assert not res.ok and res.reason.startswith("ancestor /") and "untrusted uid" in res.reason


def test_the_file_owner_alone_is_checked_when_the_chain_is_trusted(root, monkeypatch):
    # chain trusted via the anchor's own uid in the set; the file's uid is what differs
    real_fstat = os.fstat

    class FakeStat:
        def __init__(self, st, uid):
            self._st, self.st_uid = st, uid

        def __getattr__(self, name):
            return getattr(self._st, name)

    def fstat(fd):
        st = real_fstat(fd)
        import stat as _s
        return FakeStat(st, ME + 1) if _s.S_ISREG(st.st_mode) else st

    monkeypatch.setattr(ownerprot.os, "fstat", fstat)
    res = check(root)
    assert not res.ok and res.reason == f"file owned by untrusted uid {ME + 1}"


def test_a_symlinked_file_is_refused_even_to_a_good_target(root):
    target = root / "etc" / "real.yaml"
    target.write_text("a: 1\n")
    target.chmod(0o644)
    link = root / "etc" / "link.yaml"
    link.symlink_to(target)
    assert check(root, target).ok
    res = check(root, link)
    assert not res.ok and "symlink" in res.reason


def test_a_symlinked_ancestor_is_refused(root):
    real = root / "real"
    real.mkdir()
    real.chmod(0o755)
    (real / "system.yaml").write_text("a: 1\n")
    (real / "system.yaml").chmod(0o644)
    (root / "via").symlink_to(real)
    assert check(root, real / "system.yaml").ok
    res = check(root, root / "via" / "system.yaml")
    assert not res.ok and "symlink" in res.reason


@pytest.mark.parametrize("mode", [0o775, 0o757, 0o777])
def test_a_group_or_other_writable_ancestor_is_refused(root, mode):
    (root / "etc").chmod(mode)
    res = check(root)
    assert not res.ok and res.reason == "ancestor 'etc' is group- or other-writable"


def test_a_writable_anchor_is_refused(root):
    root.chmod(0o777)
    assert not check(root).ok


def test_a_fifo_a_directory_a_missing_file_and_a_bad_path_are_refused(root):
    fifo = root / "etc" / "fifo.yaml"
    os.mkfifo(fifo, 0o644)
    assert check(root, fifo).reason == "not a regular file"  # opened O_NONBLOCK: no hang
    d = root / "etc" / "dir.yaml"
    d.mkdir()
    d.chmod(0o755)
    assert check(root, d).reason == "not a regular file"
    assert check(root, root / "etc" / "nope.yaml").reason == "file does not exist"
    for bad in ("relative/path", "/a/../b", "bad\0nul"):
        assert not read_owner_file(bad, TRUSTED).ok
    assert not read_owner_file(root.parent / "x.yaml", TRUSTED, anchor=root).ok


def test_an_oversize_file_is_refused(root):
    big = root / "etc" / "big.yaml"
    big.write_bytes(b"x" * 100)
    big.chmod(0o644)
    assert not check(root, big, max_bytes=50).ok and check(root, big, max_bytes=100).ok


def test_the_real_root_is_untrusted_when_only_our_uid_is(tmp_path):
    f = tmp_path / "x.yaml"
    f.write_text("a: 1\n")
    f.chmod(0o644)
    if os.stat("/").st_uid == ME:
        pytest.skip("running as the owner of /")
    res = read_owner_file(f, TRUSTED)
    assert not res.ok and res.reason.startswith("ancestor /") and "untrusted uid" in res.reason


def test_the_check_goes_by_descriptor_a_swap_after_open_changes_nothing(root, monkeypatch):
    f = root / "etc" / "system.yaml"
    f.chmod(0o666)
    good = root / "etc" / "good.tmp"
    good.write_text("a: 1\n")
    good.chmod(0o644)
    real_open = os.open

    def swapping_open(path, flags, *a, **kw):
        fd = real_open(path, flags, *a, **kw)
        if path == "system.yaml" and not flags & os.O_DIRECTORY:
            os.rename(good, f)
        return fd

    monkeypatch.setattr(ownerprot.os, "open", swapping_open)
    res = check(root)
    assert not res.ok and res.data == b""


def test_the_bytes_returned_are_the_descriptors_not_the_swapped_path(root, monkeypatch):
    f = root / "etc" / "system.yaml"
    evil = root / "etc" / "evil.tmp"
    evil.write_text("evil: true\n")
    evil.chmod(0o666)
    real_open = os.open

    def swapping_open(path, flags, *a, **kw):
        fd = real_open(path, flags, *a, **kw)
        if path == "system.yaml" and not flags & os.O_DIRECTORY:
            os.rename(evil, f)
        return fd

    monkeypatch.setattr(ownerprot.os, "open", swapping_open)
    res = check(root)
    assert res.ok and res.data == b"a: 1\n"


def test_require_owner_file_names_the_file_and_the_failed_condition(root):
    (root / "etc" / "system.yaml").chmod(0o666)
    with pytest.raises(PreconditionError) as e:
        require_owner_file(root / "etc" / "system.yaml", TRUSTED, anchor=root)
    assert str(e.value) == f"{root / 'etc' / 'system.yaml'}: not owner-protected: file is group- or other-writable"
    (root / "etc" / "system.yaml").chmod(0o644)
    assert require_owner_file(root / "etc" / "system.yaml", TRUSTED, anchor=root) == b"a: 1\n"


# ------------------------------------------------------------------ parity with llmgateway.ownercheck


def _plant(root, name, *, mode=0o644, kind="file", content=b"a: 1\n"):
    p = root / "etc" / name
    if kind == "file":
        p.write_bytes(content)
        p.chmod(mode)
    elif kind == "symlink":
        t = root / "etc" / (name + ".target")
        t.write_bytes(content)
        t.chmod(0o644)
        p.symlink_to(t)
    elif kind == "fifo":
        os.mkfifo(p, 0o644)
    elif kind == "dir":
        p.mkdir()
        p.chmod(0o755)
    return p


CASES = [
    ("good", dict()),
    ("g-writable", dict(mode=0o664)),
    ("o-writable", dict(mode=0o646)),
    ("readonly", dict(mode=0o444)),
    ("link", dict(kind="symlink")),
    ("fifo", dict(kind="fifo")),
    ("dir", dict(kind="dir")),
    ("empty", dict(content=b"")),
    ("binary", dict(content=bytes(range(256)))),
]


@pytest.mark.parametrize("name,kw", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("trusted", [TRUSTED, OTHER], ids=["me-trusted", "me-untrusted"])
def test_it_agrees_with_the_gateway_check_on_the_same_planted_file(root, name, kw, trusted):
    from llmgateway.ownercheck import read_owner_file as gateway_read  # an unimportable llmgateway FAILS this test
    p = _plant(root, name, **kw)
    ours, theirs = read_owner_file(p, trusted, anchor=root), gateway_read(p, trusted, anchor=root)
    assert (ours.ok, ours.reason, ours.data) == (theirs.ok, theirs.reason, theirs.data)


@pytest.mark.parametrize("etc_mode", [0o755, 0o775, 0o777])
def test_it_agrees_with_the_gateway_check_on_a_writable_ancestor_and_a_missing_file(root, etc_mode):
    from llmgateway.ownercheck import read_owner_file as gateway_read
    (root / "etc").chmod(etc_mode)
    for p in (root / "etc" / "system.yaml", root / "etc" / "nope.yaml"):
        ours, theirs = read_owner_file(p, TRUSTED, anchor=root), gateway_read(p, TRUSTED, anchor=root)
        assert (ours.ok, ours.reason, ours.data) == (theirs.ok, theirs.reason, theirs.data)


def test_it_agrees_with_the_gateway_check_on_an_oversize_file_and_a_bad_path(root):
    from llmgateway.ownercheck import read_owner_file as gateway_read
    p = _plant(root, "big.yaml", content=b"x" * 100)
    for mb in (50, 100):
        ours, theirs = read_owner_file(p, TRUSTED, anchor=root, max_bytes=mb), gateway_read(p, TRUSTED, anchor=root, max_bytes=mb)
        assert (ours.ok, ours.reason, ours.data) == (theirs.ok, theirs.reason, theirs.data)
    for bad in ("relative/path", "/a/../b"):
        assert read_owner_file(bad, TRUSTED).reason == gateway_read(bad, TRUSTED).reason
