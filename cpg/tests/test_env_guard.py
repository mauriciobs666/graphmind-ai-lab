"""The test-mode safety net (``ccg.atrest.assert_not_real_credentials`` + ``tests/conftest.py``): nothing can reach the real ``~/.config/ccg``.

Guard tests use a FAKE ``$HOME`` (a tmp dir), so even a broken guard writes only there, never into the user's real directory.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ccg import atrest

CPG = Path(atrest.__file__).resolve().parent.parent


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".config").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CCG_CREDENTIALS_DIR")
    return home, home / ".config" / "ccg" / "keys"


def test_every_test_runs_in_test_mode_with_both_overrides_at_a_tmp_dir():
    assert os.environ[atrest.FORBID_REAL_ENV] == "1"
    real = Path(os.path.expanduser("~")) / ".config" / "ccg"
    for var in ("CCG_CREDENTIALS_DIR", "CCG_CONFIG_DIR"):
        v = Path(os.environ[var])
        assert v != real and real not in v.parents
    assert os.environ["CCG_CREDENTIALS_DIR"] != os.environ["CCG_CONFIG_DIR"]


@pytest.mark.parametrize("rel", ["", "keys", "keys/x.hmac", "../ccg/keys", "a/../b"])
def test_a_path_in_the_real_area_is_refused(fake_home, rel):
    home, _ = fake_home
    with pytest.raises(atrest.RealCredentialsDirTouched):
        atrest.assert_not_real_credentials(home / ".config" / "ccg" / rel)


@pytest.mark.parametrize("rel", [".config", ".config/ccg2", ".config/ccgx/keys", "elsewhere"])
def test_a_path_next_to_the_real_area_is_allowed(fake_home, rel):
    atrest.assert_not_real_credentials(fake_home[0] / rel)


def test_a_symlink_into_the_real_area_is_refused_by_realpath(fake_home, tmp_path):
    home, keys = fake_home
    (home / ".config" / "ccg").mkdir()
    link = tmp_path / "alias"
    link.symlink_to(home / ".config" / "ccg")
    with pytest.raises(atrest.RealCredentialsDirTouched):
        atrest.assert_not_real_credentials(link / "keys")


def test_without_the_marker_nothing_is_refused(fake_home, monkeypatch):
    monkeypatch.delenv(atrest.FORBID_REAL_ENV)
    atrest.assert_not_real_credentials(fake_home[1])


def _tree(home: Path) -> list[str]:
    return sorted(str(p.relative_to(home)) for p in home.rglob("*"))


def test_every_credentials_operation_refuses_the_real_area_and_writes_nothing(fake_home):
    home, keys = fake_home
    before = _tree(home)
    ops = [lambda: atrest.hmac_key("envguard", keys), lambda: atrest.key_path("envguard", keys),
           lambda: atrest.write_acl("envguard", b"x", keys), lambda: atrest.read_acl("envguard", keys),
           lambda: atrest.remove_acl("envguard", keys), lambda: atrest.sweep_stale_acl_tmp(0, keys),
           lambda: atrest.prepare_acl_target("envguard", keys), lambda: atrest.acl_path("envguard", keys),
           lambda: atrest.credentials_dir(keys),
           lambda: atrest.write_private(keys / "e.x", b"x", keys), lambda: atrest.read_private(keys / "e.x", keys),
           lambda: atrest.ensure_private(keys / "e.x", keys)]
    for op in ops:
        with pytest.raises(atrest.RealCredentialsDirTouched):
            op()
    assert _tree(home) == before


def test_the_default_area_is_refused_when_no_override_is_given(fake_home, monkeypatch):
    home, keys = fake_home
    monkeypatch.setattr(atrest, "KEY_ROOT", keys)
    with pytest.raises(atrest.RealCredentialsDirTouched):
        atrest.hmac_key("envguard")
    with pytest.raises(atrest.RealCredentialsDirTouched):
        atrest.write_acl("envguard", b"x")
    assert not (home / ".config" / "ccg").exists()


def test_a_reload_of_atrest_keeps_the_protection(fake_home):
    """Run in a child (a reload in this process would replace the exception classes other test modules hold)."""
    home, keys = fake_home
    code = ("import sys, importlib; from ccg import atrest\nimportlib.reload(atrest)\n"
            "try:\n    atrest.write_acl('envguard', b'x', sys.argv[1])\nexcept atrest.RealCredentialsDirTouched:\n    print('REFUSED')\n")
    out = subprocess.run([sys.executable, "-c", code, str(keys)], cwd=CPG, capture_output=True, text=True,
                         env={**os.environ, "HOME": str(home)}).stdout
    assert out.strip() == "REFUSED" and not (home / ".config" / "ccg").exists()


def test_a_child_process_inherits_the_protection(fake_home):
    home, keys = fake_home
    code = ("import sys; from ccg import atrest\n"
            "try:\n    atrest.write_acl('envguard', b'x', sys.argv[1])\nexcept atrest.RealCredentialsDirTouched:\n    print('REFUSED')\n")
    out = subprocess.run([sys.executable, "-c", code, str(keys)], cwd=CPG, capture_output=True, text=True,
                         env={**os.environ, "HOME": str(home)}).stdout
    assert out.strip() == "REFUSED" and not (home / ".config" / "ccg").exists()


def test_an_explicit_keys_root_that_disagrees_with_the_env_var_is_an_error_in_test_mode(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("CCG_CREDENTIALS_DIR", str(a))
    with pytest.raises(atrest.AtRestError, match="different directories"):
        atrest.credentials_dir(b)
    assert atrest.credentials_dir(a) == a and atrest.credentials_dir() == a
    monkeypatch.delenv(atrest.FORBID_REAL_ENV)
    assert atrest.credentials_dir(b) == a                      # production behaviour: the variable wins, silently, as before


def test_a_test_that_forgets_the_overrides_fails_instead_of_writing(fake_home, monkeypatch):
    home, keys = fake_home
    monkeypatch.delenv("CCG_CONFIG_DIR")
    monkeypatch.setattr(atrest, "KEY_ROOT", keys)
    with pytest.raises(atrest.RealCredentialsDirTouched):
        atrest.hmac_key("envguard")
    assert not (home / ".config" / "ccg").exists()


def test_the_creating_primitives_refuse_the_real_area_on_their_own(fake_home):
    home, keys = fake_home
    with pytest.raises(atrest.RealCredentialsDirTouched):
        atrest._private_dir(keys, keys, True)
    (home / ".config" / "ccg").mkdir()
    with pytest.raises(atrest.RealCredentialsDirTouched):
        atrest._write_new(home / ".config" / "ccg" / "x.tmp", b"x")
    assert sorted(p.name for p in (home / ".config" / "ccg").iterdir()) == []
