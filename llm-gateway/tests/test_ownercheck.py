"""Owner-protection check (plan §3.5, §5.2): real files, dirs and symlinks in a tmp dir."""

import os

import pytest

from llmgateway import ownercheck
from llmgateway.ownercheck import read_owner_file

ME = os.getuid()
TRUSTED = frozenset({ME})
OTHER = frozenset({ME + 4242})


@pytest.fixture
def root(tmp_path):
    """A trusted chain: <tmp>/root/etc/gateway.toml, all 0755/0644, owned by this uid."""
    r = tmp_path / "root"
    (r / "etc").mkdir(parents=True)
    r.chmod(0o755)
    (r / "etc").chmod(0o755)
    f = r / "etc" / "gateway.toml"
    f.write_text("a = 1\n")
    f.chmod(0o644)
    return r


def check(root, path=None, trusted=TRUSTED, **kw):
    return read_owner_file(path or root / "etc" / "gateway.toml", trusted, anchor=root, **kw)


def test_protected_file_is_read_with_key(root):
    res = check(root)
    assert res.ok and res.data == b"a = 1\n"
    st = os.stat(root / "etc" / "gateway.toml")
    assert res.key == (st.st_dev, st.st_ino, st.st_mtime_ns, st.st_size)


def test_missing_file_refused(root):
    res = check(root, root / "etc" / "nope.toml")
    assert not res.ok and "exist" in res.reason


@pytest.mark.parametrize("mode", [0o664, 0o646, 0o666, 0o620, 0o602])
def test_group_or_other_writable_file_refused(root, mode):
    f = root / "etc" / "gateway.toml"
    f.chmod(mode)
    res = check(root)
    assert not res.ok and "writable" in res.reason and res.data == b""


def test_wrong_owner_refused_via_injected_trusted_uids(root):
    res = check(root, trusted=OTHER)
    assert not res.ok and "untrusted uid" in res.reason


def test_own_uid_file_accepted_only_when_injected(root):
    assert check(root, trusted=TRUSTED).ok
    assert not check(root, trusted=OTHER).ok


def test_symlinked_file_refused_even_to_a_good_target(root):
    target = root / "etc" / "real.toml"
    target.write_text("a = 1\n")
    target.chmod(0o644)
    link = root / "etc" / "link.toml"
    link.symlink_to(target)
    assert check(root, root / "etc" / "real.toml").ok
    res = check(root, link)
    assert not res.ok and "symlink" in res.reason


def test_symlinked_ancestor_refused(root, tmp_path):
    real = root / "real"
    real.mkdir()
    real.chmod(0o755)
    (real / "gateway.toml").write_text("a = 1\n")
    (real / "gateway.toml").chmod(0o644)
    (root / "viaLink").symlink_to(real)
    assert check(root, root / "real" / "gateway.toml").ok
    res = check(root, root / "viaLink" / "gateway.toml")
    assert not res.ok and "symlink" in res.reason


@pytest.mark.parametrize("mode", [0o775, 0o757, 0o777])
def test_group_or_other_writable_ancestor_refused(root, mode):
    (root / "etc").chmod(mode)
    res = check(root)
    assert not res.ok and "ancestor" in res.reason and "writable" in res.reason


def test_writable_anchor_itself_refused(root):
    root.chmod(0o777)
    assert not check(root).ok


def test_wrong_owner_ancestor_refused(tmp_path):
    """A file of mine, trusted uid = mine, walked from the real `/`: the root ancestor fails."""
    f = tmp_path / "x.toml"
    f.write_text("a=1")
    f.chmod(0o644)
    res = read_owner_file(f, frozenset({ME}))
    if os.stat("/").st_uid == ME:
        pytest.skip("running as the owner of /")
    assert not res.ok and res.reason.startswith("ancestor /") and "untrusted uid" in res.reason


def test_root_owned_chain_accepted_when_root_trusted():
    if os.stat("/usr/bin/env").st_mode & 0o022 or os.path.realpath("/usr/bin") != "/usr/bin":
        pytest.skip("host layout differs")
    assert read_owner_file("/usr/bin/env", frozenset({0}), max_bytes=1 << 30).ok


def test_not_a_regular_file_refused(root):
    (root / "etc" / "dir.toml").mkdir()
    (root / "etc" / "dir.toml").chmod(0o755)
    res = check(root, root / "etc" / "dir.toml")
    assert not res.ok
    fifo = root / "etc" / "fifo.toml"
    os.mkfifo(fifo, 0o644)
    assert "regular" in check(root, fifo).reason


def test_oversize_refused(root):
    big = root / "etc" / "big.toml"
    big.write_bytes(b"x" * 100)
    big.chmod(0o644)
    assert not check(root, big, max_bytes=50).ok
    assert check(root, big, max_bytes=100).ok


@pytest.mark.parametrize("bad", ["relative/path", "/a/../b", "bad\0nul"])
def test_malformed_path_refused(root, bad):
    assert not read_owner_file(bad, TRUSTED).ok


def test_path_outside_anchor_refused(root, tmp_path):
    assert not read_owner_file(tmp_path / "x.toml", TRUSTED, anchor=root).ok


def test_rechecked_on_every_call_tightening_and_loosening_take_effect(root):
    f = root / "etc" / "gateway.toml"
    assert check(root).ok
    f.chmod(0o666)
    assert not check(root).ok
    f.chmod(0o644)
    assert check(root).ok
    (root / "etc").chmod(0o777)
    assert not check(root).ok


def test_no_anchor_walks_from_real_root(tmp_path):
    # tmp_path's real ancestors (/tmp ...) are not all trusted-and-unwritable: refused, not crashed.
    f = tmp_path / "x.toml"
    f.write_text("a=1")
    f.chmod(0o644)
    res = read_owner_file(f, frozenset({0, ME}))
    assert isinstance(res.ok, bool)
    assert not read_owner_file("/etc/does-not-exist-llmgw", frozenset({0})).ok


def test_checks_the_descriptor_not_the_path(root, monkeypatch):
    """Group-writable file opened, then a good file is swapped into its name: still refused."""
    f = root / "etc" / "gateway.toml"
    f.chmod(0o666)
    good = root / "etc" / "good.tmp"
    good.write_text("a = 1\n")
    good.chmod(0o644)
    real_open = os.open

    def swapping_open(path, flags, *a, **kw):
        fd = real_open(path, flags, *a, **kw)
        if path == "gateway.toml" and not flags & os.O_DIRECTORY:
            os.rename(good, f)
        return fd

    monkeypatch.setattr(ownercheck.os, "open", swapping_open)
    res = check(root)
    assert not res.ok and res.data == b""


def test_reads_the_descriptor_content_not_the_swapped_path(root, monkeypatch):
    f = root / "etc" / "gateway.toml"
    evil = root / "etc" / "evil.tmp"
    evil.write_text("evil = true\n")
    evil.chmod(0o666)
    real_open = os.open

    def swapping_open(path, flags, *a, **kw):
        fd = real_open(path, flags, *a, **kw)
        if path == "gateway.toml" and not flags & os.O_DIRECTORY:
            os.rename(evil, f)
        return fd

    monkeypatch.setattr(ownercheck.os, "open", swapping_open)
    res = check(root)
    assert res.ok and res.data == b"a = 1\n"
