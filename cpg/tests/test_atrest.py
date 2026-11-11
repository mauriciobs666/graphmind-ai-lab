"""P1-02 — bundle-at-rest and HMAC-key custody (F-1): `ccg.atrest` and the no-LLM-reads-a-bundle guard.

Decision text: `cpg/infra/README.md`, section "Bundle-at-rest". Plan: `docs/plans/connected-code-graph-p1-extraction.md`
§ P1-02. Every filesystem test runs in temporary areas (``root=`` / ``keys_root=`` / ``area=``); none touches the real
`cpg/.ccg-work/` or `~/.config/ccg/`.

Each accessor is pinned to one area: bundles to the work area, the HMAC key to the key area, ACL credentials to the
credentials dir. A symlink anywhere on the lexical path below an area root is refused (B1 of
`docs/reviews/connected-code-graph-p1-02-48.md`); the planted links below are the positive controls.

The guard (behaviour c) applies `cpg/eval/tools/audit.py`'s `pattern_escapes(root, base_dir, pat)` to every path
pattern a tool allow-list grants. See the guard section for what it reads and its stated bound.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from ccg import atrest
from ccg.atrest import AtRestError

CPG = Path(__file__).resolve().parent.parent
REPO = CPG.parent
KIT = CPG / "eval" / "headless"
SPEC5 = REPO / "docs" / "plans" / "connected-code-graph-bundle-spec5.md"


def _load_audit():
    spec = importlib.util.spec_from_file_location("audit", CPG / "eval" / "tools" / "audit.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("audit", mod)
    spec.loader.exec_module(mod)
    return mod


audit = _load_audit()


def _mode(p: Path) -> int:
    return stat.S_IMODE(p.lstat().st_mode)


@pytest.fixture
def work(tmp_path):
    root = tmp_path / ".ccg-work"
    root.mkdir()
    return root


@pytest.fixture
def keys(tmp_path):
    """A private key area outside the (real) repo tree, as ~/.config/ccg/keys is."""
    k = tmp_path / "home" / ".config" / "ccg" / "keys"
    k.parent.mkdir(parents=True, mode=0o700)
    return k


@pytest.fixture(autouse=True)
def _no_env_credentials(monkeypatch):
    monkeypatch.delenv("CCG_CREDENTIALS_DIR", raising=False)


# The published test keys and their checks, read from the spec text (never from atrest's constants).
def _spec5_test_keys() -> tuple[list[bytes], list[str]]:
    text = SPEC5.read_text()
    sec = text[text.index("## 9. Conformance vectors"):]
    para = sec[:sec.index("### 9.1")]
    keys = [bytes.fromhex(h) for h in re.findall(r"`([0-9a-f]{64})`", para)]
    eh30 = re.search(r"^\| EH-30 \| `ccg\.bindParts\.keycheck\.v1` \| \(empty\) \| `([0-9a-f]{64})` \|", sec, re.M).group(1)
    k2 = re.search(r"`bind_key_check\(K2\)` is `([0-9a-f]{64})`", sec).group(1)
    return keys, [eh30, k2]


SPEC_KEYS, SPEC_CHECKS = _spec5_test_keys()


def test_spec5_publishes_exactly_two_test_keys_with_these_checks():
    assert SPEC_KEYS == [bytes(range(32)), bytes(range(32, 64))]
    assert [atrest.bind_key_check(k) for k in SPEC_KEYS] == SPEC_CHECKS


# =============================================================================================
# (a) modes and location
# =============================================================================================

def test_bundle_dir_is_0700_under_bundles_system(work):
    d = atrest.bundle_dir("shop", root=work)
    assert d == work / "bundles" / "shop"
    assert d.is_dir()
    assert _mode(d) == 0o700
    assert _mode(work / "bundles") == 0o700


def test_written_bundle_is_0600_valid_and_read_back(work):
    p = atrest.write_bundle("shop", "u1", "b1", [_header()], root=work)
    assert p == work / "bundles" / "shop" / "u1.b1.bundle.jsonl"
    assert _mode(p) == 0o600
    from ccg import bundle
    data = atrest.read_bundle("shop", "u1", "b1", root=work)
    assert bundle.loads(data)[0]["type"] == "header"


def test_modes_are_set_even_when_the_umask_strips_owner_bits(work):
    old = os.umask(0o277)
    try:
        d = atrest.bundle_dir("shop", root=work)
        p = atrest.write_private(d / "x.bin", b"x", area=work)
    finally:
        os.umask(old)
    assert _mode(d) == 0o700 and _mode(work / "bundles") == 0o700
    assert _mode(p) == 0o600


def test_write_private_is_0600_even_with_umask_0(work):
    old = os.umask(0)
    try:
        p = atrest.write_private(work / "misc" / "x.bin", b"secret\n", area=work)
    finally:
        os.umask(old)
    assert _mode(p) == 0o600
    assert _mode(work / "misc") == 0o700


# =============================================================================================
# ensure_private: refusals
# =============================================================================================

def _private_file(p: Path, mode=0o600, data=b"x"):
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    p.write_bytes(data)
    os.chmod(p, mode)
    return p


def test_ensure_private_raises_on_world_readable_file(work):
    f = _private_file(work / "misc" / "leak.bin", 0o644)
    with pytest.raises(AtRestError, match="group/other"):
        atrest.ensure_private(f, area=work)


def test_ensure_private_raises_on_group_readable_dir(work):
    d = work / "bundles"
    d.mkdir()
    os.chmod(d, 0o750)
    with pytest.raises(AtRestError, match="group/other"):
        atrest.ensure_private(d, area=work)


def test_ensure_private_accepts_a_private_file(work):
    f = _private_file(work / "misc" / "ok.bin")
    assert atrest.ensure_private(f, area=work) == f.resolve()


def test_ensure_private_does_not_loosen_or_fix_an_exposed_file(work):
    f = _private_file(work / "misc" / "leak.bin", 0o604)
    with pytest.raises(AtRestError):
        atrest.ensure_private(f, area=work)
    assert _mode(f) == 0o604          # a raise, not a silent chmod: the exposure needs a human


def test_read_private_refuses_an_exposed_file(work):
    f = _private_file(work / "misc" / "leak.bin", 0o640)
    with pytest.raises(AtRestError, match="group/other"):
        atrest.read_private(f, area=work)


def test_path_outside_its_area_is_refused(work, tmp_path):
    outside = tmp_path / "elsewhere" / "k.bin"
    with pytest.raises(AtRestError, match="outside"):
        atrest.ensure_private(outside, area=work)
    with pytest.raises(AtRestError, match="outside"):
        atrest.write_private(outside, b"x", area=work)
    assert not outside.exists()


@pytest.mark.parametrize("rel", ["bundles/../../x", "bundles/../misc/x", "../x"])
def test_lexical_dotdot_is_refused_even_when_it_lands_inside(work, rel):
    (work / "misc").mkdir(mode=0o700)
    with pytest.raises(AtRestError, match=r"\.\."):
        atrest.ensure_private(str(work) + "/" + rel, area=work)


def test_area_root_itself_is_refused(work):
    with pytest.raises(AtRestError, match="outside"):
        atrest.ensure_private(work, area=work)


def test_generic_accessors_need_an_explicit_area(work):
    with pytest.raises(TypeError):
        atrest.read_private(work / "x")                     # no implicit "either area" default (B1)


# --- B1 positive controls: planted symlinks -------------------------------------------------

def test_bundle_named_link_into_the_key_area_is_refused(work, keys):
    """The reviewer's plant: bundles/shop/u1.b9.bundle.jsonl -> keys/shop.hmac. Key bytes never come back."""
    atrest.hmac_key("shop", keys_root=keys)
    d = atrest.bundle_dir("shop", root=work)
    (d / "u1.b9.bundle.jsonl").symlink_to(keys / "shop.hmac")
    with pytest.raises(AtRestError, match="symlink"):
        atrest.read_bundle("shop", "u1", "b9", root=work)
    with pytest.raises(AtRestError):
        atrest.read_private(d / "u1.b9.bundle.jsonl", area=work)


def test_in_area_link_to_another_in_area_file_is_refused(work):
    p = atrest.write_bundle("shop", "u1", "b1", [_header()], root=work)
    (p.parent / "u1.b2.bundle.jsonl").symlink_to(p)
    with pytest.raises(AtRestError, match="symlink"):
        atrest.read_bundle("shop", "u1", "b2", root=work)


def test_dangling_link_is_refused_and_never_written_through(work):
    d = atrest.bundle_dir("shop", root=work)
    link = d / "u1.b1.bundle.jsonl"
    link.symlink_to(d / "nowhere")
    with pytest.raises(AtRestError, match="symlink"):
        atrest.write_bundle("shop", "u1", "b1", [_header()], root=work)
    assert link.is_symlink() and not (d / "nowhere").exists()


def test_linked_directory_inside_the_area_is_refused(work, tmp_path):
    (work / "bundles").mkdir(mode=0o700)
    real = work / "bundles" / "real"
    real.mkdir(mode=0o700)
    (work / "bundles" / "shop").symlink_to(real)            # an in-area directory link
    with pytest.raises(AtRestError, match="symlink"):
        atrest.bundle_dir("shop", root=work)


def test_symlink_out_of_the_work_area_is_refused(work, tmp_path):
    target = tmp_path / "outside"
    target.mkdir(mode=0o700)
    (work / "bundles").symlink_to(target)
    with pytest.raises(AtRestError):
        atrest.bundle_dir("shop", root=work)
    assert not (target / "shop").exists()


def test_key_file_replaced_by_a_link_is_refused(keys, tmp_path):
    atrest.hmac_key("shop", keys_root=keys)
    p = atrest.key_path("shop", keys_root=keys)
    other = _private_file(tmp_path / "other.hex", data=os.urandom(32).hex().encode())
    p.unlink()
    p.symlink_to(other)
    with pytest.raises(AtRestError, match="symlink"):
        atrest.hmac_key("shop", keys_root=keys)


def test_read_checks_the_opened_file_not_the_name(work, monkeypatch):
    """O_NOFOLLOW + fstat: the mode check is on the fd that is read, so a swap after the name check is caught."""
    f = _private_file(work / "misc" / "x.bin")
    real_open = os.open

    def swapping_open(path, flags, *a, **kw):
        if os.path.basename(str(path)) == f.name and not (flags & os.O_WRONLY):      # the leaf open
            os.chmod(f, 0o644)                              # exposed between the name check and the open
        return real_open(path, flags, *a, **kw)

    monkeypatch.setattr(atrest.os, "open", swapping_open)
    with pytest.raises(AtRestError, match="group/other"):
        atrest.read_private(f, area=work)


def test_read_never_follows_a_link_swapped_in_after_the_name_check(work, tmp_path, monkeypatch):
    """O_NOFOLLOW: the file is replaced by a link to a secret between the check and the open."""
    f = _private_file(work / "misc" / "x.bin", data=b"bundle")
    secret = _private_file(tmp_path / "secret.bin", data=b"KEY-BYTES")
    real_open = os.open

    def swapping_open(path, flags, *a, **kw):
        if os.path.basename(str(path)) == f.name and not (flags & os.O_WRONLY):      # the leaf open
            os.unlink(f)
            os.symlink(secret, f)
        return real_open(path, flags, *a, **kw)

    monkeypatch.setattr(atrest.os, "open", swapping_open)
    with pytest.raises(AtRestError, match="without following links"):
        atrest.read_private(f, area=work)


def test_read_never_follows_a_parent_directory_swapped_for_a_link(work, tmp_path, monkeypatch):
    """P2-4: a parent directory replaced by a link between the name check and the open is not followed."""
    f = _private_file(work / "misc" / "x.bin", data=b"bundle")
    elsewhere = tmp_path / "elsewhere"
    _private_file(elsewhere / "x.bin", data=b"KEY-BYTES")
    real_check = atrest.ensure_private

    def check_then_swap(path, area):
        r = real_check(path, area)
        os.rename(work / "misc", work / "misc.moved")
        os.symlink(elsewhere, work / "misc")
        return r

    monkeypatch.setattr(atrest, "ensure_private", check_then_swap)
    with pytest.raises(AtRestError, match="without following links"):
        atrest.read_private(f, area=work)


@pytest.mark.parametrize("bad", ["", "..", ".", "a/b", "../x", "-x", ".hidden", "a b", "x\0y", "s" * 65, "a\nb"])
def test_bad_system_names_are_refused_before_any_path_exists(work, bad):
    with pytest.raises(AtRestError):
        atrest.bundle_dir(bad, root=work)
    with pytest.raises(AtRestError):
        atrest.key_path(bad)
    assert not (work / "bundles").exists()


def test_default_areas_bundles_in_the_work_area_keys_outside_the_repo(monkeypatch):
    monkeypatch.delenv(atrest.FORBID_REAL_ENV)                 # path arithmetic only: nothing is created in the real area
    assert atrest.WORK_ROOT == CPG / ".ccg-work"
    assert atrest.KEY_ROOT == Path.home() / ".config" / "ccg" / "keys"
    assert atrest.key_path("shop") == atrest.KEY_ROOT / "shop.hmac"
    assert REPO.resolve() not in atrest.key_path("shop").resolve().parents


# =============================================================================================
# (b) retention
# =============================================================================================

def test_retention_keeps_last_good_and_in_progress_only(work):
    h = [_header()]
    old = atrest.write_bundle("shop", "u1", "b1", h, root=work)
    good = atrest.write_bundle("shop", "u1", "b2", h, root=work)
    other_unit = atrest.write_bundle("shop", "u2", "b1", h, root=work)
    building = _private_file(work / "bundles" / "shop" / "u1.b3.bundle.jsonl.tmp", data=b"partial")
    stray = _private_file(work / "bundles" / "shop" / "u1.not.a.build.bundle.jsonl")
    removed = atrest.prune_unit("shop", "u1", keep_build="b2", root=work)
    assert removed == [old]
    assert not old.exists()
    assert good.exists() and other_unit.exists() and building.exists()
    assert stray.exists()                 # not <unit>.<build>: never this unit's to delete


def test_retention_refuses_a_keep_that_does_not_exist(work):
    atrest.write_bundle("shop", "u1", "b1", [_header()], root=work)
    with pytest.raises(AtRestError, match="no bundle"):
        atrest.prune_unit("shop", "u1", keep_build="b9", root=work)
    assert (work / "bundles" / "shop" / "u1.b1.bundle.jsonl").exists()


def test_retention_does_not_match_a_unit_with_the_same_prefix(work):
    h = [_header()]
    atrest.write_bundle("shop", "u1", "b2", h, root=work)
    sibling = atrest.write_bundle("shop", "u1x", "b1", h, root=work)
    atrest.prune_unit("shop", "u1", keep_build="b2", root=work)
    assert sibling.exists()


# =============================================================================================
# (d) the per-system HMAC key
# =============================================================================================

def test_hmac_key_generated_once_0600_and_stable(keys):
    k1 = atrest.hmac_key("shop", keys_root=keys)
    p = atrest.key_path("shop", keys_root=keys)
    assert p == keys / "shop.hmac"
    assert len(k1) == 32
    assert _mode(p) == 0o600 and _mode(p.parent) == 0o700
    assert atrest.hmac_key("shop", keys_root=keys) == k1
    assert atrest.hmac_key("shop2", keys_root=keys) != k1
    assert sorted(x.name for x in keys.iterdir()) == ["shop.hmac", "shop2.hmac"]    # no temp left behind


def test_hmac_key_is_32_bytes_from_the_csprng(keys, monkeypatch):
    got = []
    real = atrest.secrets.token_bytes

    def spy(n=None):
        b = real(n)
        got.append((n, b))
        return b

    monkeypatch.setattr(atrest.secrets, "token_bytes", spy)
    key = atrest.hmac_key("shop", keys_root=keys)
    assert [b for n, b in got if n == 32] == [key]      # the key is exactly one 32-byte CSPRNG draw


def test_hmac_key_first_write_is_atomic(keys, monkeypatch):
    """m4: a crash mid-write leaves no key file (never an empty one that is 'malformed' forever)."""
    real_write = os.write

    def dying_write(fd, data):
        raise OSError("disk full")

    monkeypatch.setattr(atrest.os, "write", dying_write)
    with pytest.raises(OSError):
        atrest.hmac_key("shop", keys_root=keys)
    monkeypatch.setattr(atrest.os, "write", real_write)
    assert not atrest.key_path("shop", keys_root=keys).exists()
    assert [x.name for x in keys.iterdir()] == []
    assert len(atrest.hmac_key("shop", keys_root=keys)) == 32


def test_hmac_key_lost_race_reads_the_winner(keys, monkeypatch):
    """A concurrent first call that wins the link keeps its key; the loser reads it and leaves no temp."""
    winner = os.urandom(32)
    real_link = os.link

    def racing_link(src, dst, *a, **kw):
        _private_file(Path(dst), data=winner.hex().encode() + b"\n")
        return real_link(src, dst, *a, **kw)                # now FileExistsError

    monkeypatch.setattr(atrest.os, "link", racing_link)
    assert atrest.hmac_key("shop", keys_root=keys) == winner
    assert [x.name for x in keys.iterdir()] == ["shop.hmac"]


def test_hmac_key_exposed_file_is_refused(keys):
    atrest.hmac_key("shop", keys_root=keys)
    p = atrest.key_path("shop", keys_root=keys)
    os.chmod(p, 0o644)
    with pytest.raises(AtRestError):
        atrest.hmac_key("shop", keys_root=keys)


def test_hmac_key_is_not_printed(keys, capsys, caplog):
    k = atrest.hmac_key("shop", keys_root=keys)
    out = capsys.readouterr()
    assert k.hex() not in out.out + out.err + caplog.text


def test_hmac_key_malformed_file_is_refused(keys):
    p = atrest.key_path("shop", keys_root=keys)
    atrest.write_private(p, b"not-hex\n", area=keys)
    with pytest.raises(AtRestError, match="malformed"):
        atrest.hmac_key("shop", keys_root=keys)


@pytest.mark.parametrize("i", [0, 1])
def test_each_published_test_key_is_refused_on_load(keys, i):
    """S5-7 / M3: both -bundle-spec5.md §9 keys (checks EH-30 and §9.4's K2) never load as a system key."""
    p = atrest.key_path("shop", keys_root=keys)
    atrest.write_private(p, SPEC_KEYS[i].hex().encode() + b"\n", area=keys)
    with pytest.raises(AtRestError, match="published test key"):
        atrest.hmac_key("shop", keys_root=keys)
    assert len(atrest.hmac_key("shop2", keys_root=keys)) == 32     # positive control: a fresh key loads


@pytest.mark.parametrize("i", [0, 1])
def test_each_generated_test_key_is_refused_and_never_written(keys, monkeypatch, i):
    monkeypatch.setattr(atrest.secrets, "token_bytes", lambda n: SPEC_KEYS[i])
    with pytest.raises(AtRestError, match="published test key"):
        atrest.hmac_key("shop", keys_root=keys)
    assert not atrest.key_path("shop", keys_root=keys).exists()
    assert [x.name for x in keys.iterdir()] == []


def test_refused_checks_are_exactly_the_spec5_checks():
    assert set(atrest.PUBLISHED_TEST_KEYCHECKS) == set(SPEC_CHECKS)


def test_key_area_inside_the_repo_is_refused(tmp_path, monkeypatch):
    # A fake repo layout, so even a mutant that lets it through writes only into tmp_path.
    fake_repo = tmp_path / "repo"
    (fake_repo / "cpg" / ".ccg-work").mkdir(parents=True)
    monkeypatch.setattr(atrest, "REPO_ROOT", fake_repo)
    inside = fake_repo / "cpg" / ".ccg-work" / "keys"
    with pytest.raises(AtRestError, match="inside the repository"):
        atrest.hmac_key("shop", keys_root=inside)
    assert not inside.exists()


def test_the_real_repo_is_never_a_key_area():
    with pytest.raises(AtRestError, match="inside the repository"):
        atrest.key_path("shop", keys_root=CPG / ".ccg-work" / "keys")


def test_key_area_accessors_refuse_work_area_paths_and_back(work, keys):
    with pytest.raises(AtRestError, match="outside"):
        atrest.write_private(keys / "x.bundle.jsonl", b"x", area=work)
    with pytest.raises(AtRestError, match="outside"):
        atrest.write_private(work / "keys" / "shop.hmac", b"x", area=keys)


# =============================================================================================
# (e) ACL credentials
# =============================================================================================

def test_acl_path_default_and_env_override(keys, tmp_path, monkeypatch):
    monkeypatch.delenv(atrest.FORBID_REAL_ENV)                 # path arithmetic only, and the env-wins rule of production
    assert atrest.acl_path("shop", keys_root=keys) == keys / "shop.acl"
    assert atrest.acl_path("shop") == atrest.KEY_ROOT / "shop.acl"
    alt = tmp_path / "creds"
    monkeypatch.setenv("CCG_CREDENTIALS_DIR", str(alt))
    assert atrest.acl_path("shop", keys_root=keys) == alt / "shop.acl"


def test_acl_round_trip_through_the_env_override(tmp_path, monkeypatch):
    """M2: the override is the case CCG_CREDENTIALS_DIR exists for, so it must read and write."""
    alt = tmp_path / "creds"
    monkeypatch.setenv("CCG_CREDENTIALS_DIR", str(alt))
    p = atrest.write_acl("shop", b"ccg_build_shop pw\n")
    assert p == alt / "shop.acl"
    assert _mode(p) == 0o600 and _mode(alt) == 0o700
    assert atrest.read_acl("shop") == b"ccg_build_shop pw\n"
    os.chmod(p, 0o640)
    with pytest.raises(AtRestError):
        atrest.read_acl("shop")


def test_acl_round_trip_default_key_area(keys):
    atrest.write_acl("shop", b"x\n", keys_root=keys)
    assert atrest.read_acl("shop", keys_root=keys) == b"x\n"
    assert _mode(keys / "shop.acl") == 0o600


def test_credentials_dir_inside_the_repo_is_refused(monkeypatch):
    monkeypatch.setenv("CCG_CREDENTIALS_DIR", str(CPG / ".ccg-work" / "keys"))
    with pytest.raises(AtRestError, match="inside the repository"):
        atrest.acl_path("shop")
    with pytest.raises(AtRestError, match="inside the repository"):
        atrest.write_acl("shop", b"x")
    assert not (CPG / ".ccg-work" / "keys").exists()


# =============================================================================================
# gitignore: the work area never reaches git
# =============================================================================================

@pytest.mark.parametrize("rel", ["cpg/.ccg-work/bundles/shop/u.b.bundle.jsonl", "cpg/.ccg-work/x.tmp",
                                 "cpg/eval/shop.hmac", "cpg/shop.acl"])
def test_work_area_paths_are_gitignored(rel):
    r = subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", rel], capture_output=True)
    assert r.returncode == 0, f"{rel} is not gitignored"


def test_no_work_area_file_is_tracked():
    r = subprocess.run(["git", "-C", str(REPO), "ls-files", "cpg/.ccg-work"], capture_output=True, text=True, check=True)
    assert r.stdout == ""


# =============================================================================================
# (c) the guard: no LLM-backed tool allow-list can name the bundle dir
#
# What it reads (each source that exists): the headless settings (permissions allow/ask/additionalDirectories)
# and any extra settings file a launcher passes; agents.json tool rules; every launcher flag that grants tools
# or paths, in every spelling (--allowedTools / --allowed-tools / --tools, space or `=`, --add-dir,
# --settings, --mcp-config); the `.mcp.json` `ccg` entry (every string value); `cpg/clients/mcp/server.py`
# (every MCP tool parameter outside SAFE_TOOL_PARAMS, every path-like string default, and any reference to the
# bundle store in the module). It is a static BOUND, not proof: a tool body that reaches bundles through an
# import chain, or a flag spelling the CLI adds later, is outside it (`cpg/infra/README.md`, Bundle-at-rest).
# =============================================================================================

PATH_TOOLS = ("Read", "Edit", "Write", "Glob", "Grep", "NotebookEdit", "MultiEdit")
TOOL_FLAGS = ("--allowedTools", "--allowed-tools", "--tools")
# The ccg tools' parameters (index D6: ask, resolve, refresh, raw_query) name graph objects, never files. Any
# other parameter name fails closed until it is reviewed and added here.
SAFE_TOOL_PARAMS = {"request", "anchor", "system", "scope", "cypher", "depth", "depth_bound", "use_case"}
# Exact launch strings of the `.mcp.json` `ccg` entry that P1-27 adds under review; empty until then.
MCP_LAUNCH_ALLOWLIST: set[str] = set()
BUNDLE_STORE_NAMES = {"read_bundle", "bundle_dir", "bundle_path", "prune_unit", "read_private", "atrest"}
NUL = "\0"                                            # pattern_escapes(NUL) is True: fail closed
# P2-2: the reviewed launch argv (run-writer.sh's single ARGV=( ... ) block). A flag not listed here fails
# closed; a listed flag with a fixed value must carry exactly that value; the REQUIRED flags must be present.
# `-i`/`--kill-after` belong to `env`/`timeout`, which share the block. Changing this table is a review event.
ARGV_FLAGS_ANY = {"-i", "--kill-after", "-p", "--restricted", "--strict-mcp-config", "--disable-slash-commands",
                  "--tools", "--allowedTools", "--disallowedTools", "--settings", "--agents", "--agent", "--model",
                  "--permission-mode", "--permission-prompts", "--output-format", "--verbose"}
ARGV_FIXED = {"--permission-mode": ["dontAsk"], "--permission-prompts": ["none"], "--agents": ["@agents.json"]}
ARGV_REQUIRED = {"--strict-mcp-config": None, "--permission-mode": None,
                 "--disallowedTools": {"Bash", "WebFetch", "WebSearch", "Agent"}}


def _rule_pattern(rule: str, settings_dir: Path) -> tuple[str, str] | None:
    """`Tool(spec)` -> (pattern, base_dir) under Claude Code's permission-rule path syntax; a bare rule -> None.

    `//x` is absolute, `~/x` is home (pattern_escapes refuses `~`: fails closed), `/x` is relative to the settings
    file, anything else to the session's working directory, which for this repo's agents is the repo root.
    A non-path tool's rule (`Bash(...)`, `WebFetch(domain:...)`) is returned as an escape: no such rule may be
    allowed to an LLM-backed tool here at all."""
    if "(" not in rule:
        return None
    tool, _, rest = rule.partition("(")
    spec = rest[:-1] if rest.endswith(")") else NUL
    if tool not in PATH_TOOLS:
        return (NUL, str(REPO))
    if spec.startswith("//"):
        return (spec[1:], "/")
    if spec.startswith("/"):
        return (spec[1:], str(settings_dir))
    return (spec, str(REPO))


def _launcher_vars(text: str, here: Path) -> dict[str, str]:
    """`NAME="$HERE/x"` assignments in the launcher, resolved against its directory (the only form trusted)."""
    out = {}
    for m in re.finditer(r'^([A-Z_]+)="\$HERE/([^"$]+)"', text, re.M):
        out[m.group(1)] = str(here / m.group(2))
    return out


def _launcher_tokens(text: str) -> list[str]:
    import shlex
    toks: list[str] = []
    for line in text.splitlines():
        try:
            toks += shlex.split(line, comments=True)
        except ValueError:
            toks += line.split()
    return toks


def _settings_patterns(settings: Path, src: str, rule, pats):
    s = json.loads(settings.read_text())
    perms = s.get("permissions", {}) or {}
    for key in ("allow", "ask"):
        for r in perms.get(key, []) or []:
            rule(f"{src}:{key}", r, settings.parent)
    for d in perms.get("additionalDirectories", []) or []:
        pats.append((f"{src}:additionalDirectories", str(d).rstrip("/") + "/**", str(REPO)))


def _collect(settings: Path, agents: Path, launcher: Path, mcp_json: Path, server_py: Path):
    """Every (source, pattern, base_dir) a tool allow-list grants, plus the bare path-tool grants."""
    pats: list[tuple[str, str, str]] = []
    bare: set[str] = set()

    def rule(src, r, sdir):
        got = _rule_pattern(r, sdir)
        if got is None:
            if r in PATH_TOOLS:
                bare.add(r)
        else:
            pats.append((src, *got))

    _settings_patterns(settings, settings.name, rule, pats)
    for name, a in json.loads(agents.read_text()).items():
        for r in a.get("tools", []) or []:
            rule(f"{agents.name}:{name}", r, agents.parent)

    text = launcher.read_text()
    lvars = _launcher_vars(text, launcher.parent)
    toks = _launcher_tokens(text)

    def value_path(v: str) -> str | None:
        m = re.fullmatch(r"\$\{?([A-Z_]+)\}?", v)
        return lvars.get(m.group(1)) if m else (v if "$" not in v else None)

    pats.extend(_argv_findings(text, launcher, lvars))

    i = 0
    while i < len(toks):
        t = toks[i]
        flag, eq, val = t.partition("=")
        src = f"{launcher.name}:{flag}"
        if flag in TOOL_FLAGS:
            vals = [val] if eq else []
            j = i + 1
            if not eq:
                while j < len(toks) and not toks[j].startswith("-"):
                    vals.append(toks[j]); j += 1
            for v in vals:
                for r in re.findall(r"[A-Za-z]+\([^)]*\)?|[A-Za-z]+", v):
                    rule(src, r, launcher.parent)
            i = j if not eq else i + 1
            continue
        if flag in ("--add-dir", "--settings", "--mcp-config"):
            v = val if eq else (toks[i + 1] if i + 1 < len(toks) else "")
            p = value_path(v)
            if flag == "--add-dir":
                pats.append((src, (p.rstrip("/") + "/**") if p else NUL, str(REPO)))
            elif flag == "--mcp-config":
                pats.append((src, NUL, str(REPO)))          # a headless writer gets no MCP server at all
            elif p is None or not Path(p).is_file():
                pats.append((src, NUL, str(REPO)))          # an extra settings file that cannot be read statically
            else:
                _settings_patterns(Path(p), f"{launcher.name}:--settings", rule, pats)
            i += 1 if eq else 2
            continue
        i += 1

    if mcp_json.exists():
        ccg = json.loads(mcp_json.read_text()).get("mcpServers", {}).get("ccg")
        if ccg is not None:
            vals = [ccg.get("command", "")] + list(ccg.get("args", []) or []) + list((ccg.get("env", {}) or {}).values())
            vals += [ccg.get("url", ""), ccg.get("cwd", "")]
            for v in vals:
                if isinstance(v, str) and v and v not in MCP_LAUNCH_ALLOWLIST and re.search(r"[/~.$]", v):
                    pats.append(("mcp.json:ccg", v, str(REPO)))
    if server_py.exists():
        pats.extend(_mcp_tool_patterns(server_py))
    return pats, bare


def _argv_findings(text: str, launcher: Path, lvars: dict[str, str]):
    """Judge the launcher's one ARGV=( ... ) block against ARGV_FLAGS_ANY / ARGV_FIXED / ARGV_REQUIRED."""
    out = []
    src = f"{launcher.name}:argv"
    blocks = re.findall(r"^\s*ARGV=\(\n(.*?)^\s*\)\s*$", text, re.M | re.S)
    if len(blocks) != 1:
        return [(f"{src}:block", NUL, str(REPO))]
    toks = _launcher_tokens(blocks[0])
    seen: dict[str, list[str]] = {}
    i = 0
    while i < len(toks):
        t = toks[i]
        if not t.startswith("-"):
            i += 1
            continue
        flag, eq, val = t.partition("=")
        vals = [val] if eq else []
        j = i + 1
        if not eq:
            while j < len(toks) and not toks[j].startswith("-"):
                vals.append(toks[j]); j += 1
        seen.setdefault(flag, []).extend(vals)
        if flag not in ARGV_FLAGS_ANY:
            out.append((f"{src}:{flag}", NUL, str(REPO)))
        elif flag in ARGV_FIXED:
            want = ARGV_FIXED[flag]
            got = vals[:1]
            if want == ["@agents.json"]:
                m = re.fullmatch(r"\$\{?([A-Z_]+)\}?", got[0]) if got else None
                ok = bool(m) and lvars.get(m.group(1)) == str(launcher.parent / "agents.json")
            else:
                ok = got == want
            if not ok:
                out.append((f"{src}:{flag}", NUL, str(REPO)))
        i = j if not eq else i + 1
    for flag, need in ARGV_REQUIRED.items():
        if flag not in seen or (need is not None and not need <= set(seen[flag])):
            out.append((f"{src}:{flag}", NUL, str(REPO)))
    return out


def _mcp_tool_patterns(server_py: Path):
    """Fail closed on: a tool parameter outside SAFE_TOOL_PARAMS (any type), a path-like string default, and any
    reference in the module to the bundle store (atrest's bundle readers, a `bundles`/`ccg-work` literal)."""
    import ast
    out = []
    tree = ast.parse(server_py.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in BUNDLE_STORE_NAMES or \
           isinstance(node, ast.Attribute) and node.attr in BUNDLE_STORE_NAMES or \
           isinstance(node, (ast.Import, ast.ImportFrom)) and any(a.name in BUNDLE_STORE_NAMES or a.name.endswith(".atrest")
                                                                  for a in node.names) or \
           isinstance(node, ast.ImportFrom) and (node.module or "").endswith("atrest") or \
           isinstance(node, ast.Constant) and isinstance(node.value, str) and re.search(r"bundles|ccg-work", node.value):
            out.append((f"server.py:line {getattr(node, 'lineno', '?')} (bundle store)", NUL, str(REPO)))
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any("tool" in ast.unparse(d) for d in fn.decorator_list):
            continue
        args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
        defaults = [None] * (len(fn.args.posonlyargs + fn.args.args) - len(fn.args.defaults)) + list(fn.args.defaults)
        defaults += list(fn.args.kw_defaults)
        for a in ([fn.args.vararg] if fn.args.vararg else []) + ([fn.args.kwarg] if fn.args.kwarg else []):
            out.append((f"server.py:{fn.name}(*{a.arg})", NUL, str(REPO)))
        for a, dflt in zip(args, defaults):
            if a.arg in ("self", "ctx"):
                continue
            if a.arg not in SAFE_TOOL_PARAMS:
                out.append((f"server.py:{fn.name}({a.arg})", NUL, str(REPO)))
            if isinstance(dflt, ast.Constant) and isinstance(dflt.value, str) and re.search(r"[/~]", dflt.value):
                out.append((f"server.py:{fn.name}({a.arg}=)", dflt.value, str(REPO)))
    return out


def _escapes(pats):
    root = str(KIT.resolve())
    return [(src, p) for src, p, base in pats if audit.pattern_escapes(root, base, p)]


def _guard_hook_covers(settings: Path, tools: set[str]) -> bool:
    """A bare path-tool grant is contained only by the kit's fail-closed CCG_GUARD PreToolUse hook."""
    s = json.loads(settings.read_text())
    for entry in (s.get("hooks", {}) or {}).get("PreToolUse", []) or []:
        matched = set(str(entry.get("matcher", "")).split("|"))
        cmds = " ".join(h.get("command", "") for h in entry.get("hooks", []) or [])
        if tools <= matched and "CCG_GUARD" in cmds and "exit 2" in cmds:
            return True
    return False


def _real_sources():
    return (KIT / "settings.json", KIT / "agents.json", KIT / "run-writer.sh", REPO / ".mcp.json",
            CPG / "clients" / "mcp" / "server.py")


def test_guard_real_allow_lists_stay_inside_the_kit_root():
    pats, bare = _collect(*_real_sources())
    assert _escapes(pats) == []
    assert bare, "the launcher grants bare path tools; the hook test below must see them"
    assert _guard_hook_covers(KIT / "settings.json", bare), "a bare path-tool grant without the fail-closed guard hook"


def test_guard_reads_the_real_launchers_settings_flag():
    pats, _ = _collect(*_real_sources())
    # run-writer.sh passes --settings "$SETTINGS_FILE" = $HERE/settings.json; it resolves and is read, not voided
    assert not [s for s, p, _ in pats if s.endswith("--settings") and p == NUL]


def _planted(tmp_path, rules):
    s = json.loads((KIT / "settings.json").read_text())
    s.setdefault("permissions", {})["allow"] = list(rules)
    p = tmp_path / "settings.json"
    p.write_text(json.dumps(s))
    others = _real_sources()[1:]
    return _collect(p, *others)


def test_guard_positive_control_read_cpg_glob_is_flagged(tmp_path):
    pats, _ = _planted(tmp_path, ["Read(cpg/**)"])
    assert ("settings.json:allow", "cpg/**") in _escapes(pats)


def test_guard_positive_control_read_work_area_is_flagged(tmp_path):
    pats, _ = _planted(tmp_path, ["Read(cpg/.ccg-work/**)"])
    assert ("settings.json:allow", "cpg/.ccg-work/**") in _escapes(pats)


@pytest.mark.parametrize("rule", ["Read(//etc/**)", "Read(~/x/**)", "Glob(/../../**)", "Grep(cpg/eval/**)",
                                  "Bash(cat:*)", "Read(cpg/eval/headless/../../.ccg-work/**)", "Read(cpg/eval/headless/**"])
def test_guard_any_other_pattern_outside_the_root_fails_closed(tmp_path, rule):
    pats, _ = _planted(tmp_path, [rule])
    assert _escapes(pats), f"{rule} was not flagged"


def test_guard_a_pattern_inside_the_kit_root_is_not_flagged(tmp_path):
    pats, _ = _planted(tmp_path, ["Read(cpg/eval/headless/**)", f"Read(/{KIT}/README.md)"])
    assert _escapes(pats) == []
    # `/x` is relative to the settings file's directory: the real settings file sits in the kit root.
    assert _escapes([("settings.json:allow", *_rule_pattern("Read(/README.md)", KIT))]) == []
    assert _escapes([("settings.json:allow", *_rule_pattern("Read(/../README.md)", KIT))]) != []


def test_guard_bare_grant_without_the_hook_is_flagged(tmp_path):
    s = json.loads((KIT / "settings.json").read_text())
    s["hooks"] = {}
    p = tmp_path / "settings.json"
    p.write_text(json.dumps(s))
    assert not _guard_hook_covers(p, {"Read"})


# --- Appendix A of docs/reviews/connected-code-graph-p1-02-48.md, as the test (M5) ------------

LAUNCHER_PLANTS = {   # case: (planted line, the source that must carry the escape)
    "allowedTools(space)": ('--allowedTools "Read(cpg/**)"', "--allowedTools"),
    "allowed-tools(dash)": ('--allowed-tools "Read(cpg/**)"', "--allowed-tools"),
    "allowedTools=": ('--allowedTools=Read(cpg/**)', "--allowedTools"),
    "allowed-tools=": ('"--allowed-tools=Read(cpg/.ccg-work/**)"', "--allowed-tools"),
    "tools=": ('--tools=Read,Glob(cpg/**)', "--tools"),
    "settings-extra": ('--settings /x/other.json', "--settings"),
    "settings-var": ('--settings "$OTHER_SETTINGS"', "--settings"),
    "mcp-config": ('--mcp-config /x/mcp.json', "--mcp-config"),
    "add-dir": ('--add-dir cpg/.ccg-work', "--add-dir"),
    "add-dir=": ('--add-dir=/home/u/repo/cpg', "--add-dir"),
}


def _planted_launcher(tmp_path, line):
    """The real launcher plus one line, beside a copy of the real settings, so `$HERE/settings.json` resolves
    and only the planted line can be flagged."""
    (tmp_path / "settings.json").write_text((KIT / "settings.json").read_text())
    launcher = tmp_path / "run-writer.sh"
    launcher.write_text((KIT / "run-writer.sh").read_text() + f"\n\t\t{line}\n")
    return launcher


def test_guard_planted_launcher_without_a_plant_is_clean(tmp_path):
    launcher = _planted_launcher(tmp_path, "")
    pats, _ = _collect(KIT / "settings.json", KIT / "agents.json", launcher, tmp_path / "none.json", tmp_path / "none.py")
    assert _escapes(pats) == []


@pytest.mark.parametrize("case", sorted(LAUNCHER_PLANTS))
def test_guard_every_launcher_flag_spelling_is_judged(tmp_path, case):
    line, flag = LAUNCHER_PLANTS[case]
    launcher = _planted_launcher(tmp_path, line)
    pats, _ = _collect(KIT / "settings.json", KIT / "agents.json", launcher, tmp_path / "none.json", tmp_path / "none.py")
    assert [s for s, _ in _escapes(pats) if s.startswith(f"run-writer.sh:{flag}")], f"{case} was not flagged"


# --- P2-2: the launch argv is judged against a reviewed flag allow-list --------------------------
# Each case edits the real ARGV=( ... ) block: (old text, new text, the argv flag whose finding must appear).
ARGV_PLANTS = {
    "dangerously-skip-permissions": ("--verbose", "--verbose --dangerously-skip-permissions", "--dangerously-skip-permissions"),
    "permission-mode bypass": ("--permission-mode dontAsk", "--permission-mode bypassPermissions", "--permission-mode"),
    "permission-mode acceptEdits": ("--permission-mode dontAsk", "--permission-mode acceptEdits", "--permission-mode"),
    "permission-mode=": ("--permission-mode dontAsk", "--permission-mode=bypassPermissions", "--permission-mode"),
    "permission-mode missing": ("--permission-mode dontAsk ", "", "--permission-mode"),
    "agents other file": ('--agents "$AGENTS_FILE"', "--agents /x/other.json", "--agents"),
    "plugin-dir": ("--verbose", "--verbose --plugin-dir /x/plugins", "--plugin-dir"),
    "disallowedTools removed": ("\t\t--disallowedTools Bash WebFetch WebSearch Agent\n", "", "--disallowedTools"),
    "disallowedTools narrowed": ("--disallowedTools Bash WebFetch WebSearch Agent", "--disallowedTools Bash WebFetch", "--disallowedTools"),
    "strict-mcp-config removed": (" --strict-mcp-config", "", "--strict-mcp-config"),
    "permission-prompts changed": ("--permission-prompts none", "--permission-prompts ask", "--permission-prompts"),
    "unknown flag": ("--verbose", "--verbose --add-plugin x", "--add-plugin"),
}


def _argv_planted(tmp_path, old, new):
    (tmp_path / "settings.json").write_text((KIT / "settings.json").read_text())
    (tmp_path / "agents.json").write_text((KIT / "agents.json").read_text())
    text = (KIT / "run-writer.sh").read_text()
    block = re.search(r"^\s*ARGV=\(\n.*?^\s*\)\s*$", text, re.M | re.S).group(0)
    assert block.count(old) == 1, f"stale plant: {old!r}"
    launcher = tmp_path / "run-writer.sh"
    launcher.write_text(text.replace(block, block.replace(old, new)))
    return launcher


def _argv_sources(launcher, tmp_path):
    pats, _ = _collect(KIT / "settings.json", KIT / "agents.json", launcher, tmp_path / "none.json", tmp_path / "none.py")
    return [s for s, _ in _escapes(pats) if s.startswith("run-writer.sh:argv")]


def test_guard_argv_clean_launcher_has_no_argv_finding(tmp_path):
    launcher = _argv_planted(tmp_path, "--verbose", "--verbose")
    assert _argv_sources(launcher, tmp_path) == []


@pytest.mark.parametrize("case", sorted(ARGV_PLANTS))
def test_guard_argv_flag_outside_the_reviewed_allow_list_fails_closed(tmp_path, case):
    old, new, flag = ARGV_PLANTS[case]
    launcher = _argv_planted(tmp_path, old, new)
    assert f"run-writer.sh:argv:{flag}" in _argv_sources(launcher, tmp_path), f"{case} was not flagged"


def test_guard_argv_block_missing_or_doubled_fails_closed(tmp_path):
    text = (KIT / "run-writer.sh").read_text()
    for variant in (text.replace("ARGV=(", "ARGS=("), text + "\nx() {\n\tARGV=(\n\t\tfoo\n\t)\n}\n"):
        launcher = tmp_path / "run-writer.sh"
        launcher.write_text(variant)
        assert "run-writer.sh:argv:block" in _argv_sources(launcher, tmp_path)


def test_guard_extra_settings_file_is_read_and_its_rules_judged(tmp_path):
    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps({"permissions": {"allow": ["Read(cpg/**)"]}}))
    launcher = _planted_launcher(tmp_path, f"--settings {extra}")
    pats, _ = _collect(KIT / "settings.json", KIT / "agents.json", launcher, tmp_path / "none.json", tmp_path / "none.py")
    assert ("run-writer.sh:--settings:allow", "cpg/**") in _escapes(pats)


SERVER_PLANTS = {
    "param path": "def dump(path: str) -> str:\n    return ''\n",
    "param target": "def read(target: str) -> str:\n    return ''\n",
    "param untyped": "def read(x):\n    return ''\n",
    "kwargs": "def ask(request: dict, **kw) -> dict:\n    return {}\n",
    "default path": "def peek(system: str, scope: str = 'cpg/.ccg-work/bundles/x') -> str:\n    return ''\n",
    "reads bundles": "def ask(request: dict) -> dict:\n    from ccg import atrest\n    return atrest.read_bundle('s', 'u', 'b')\n",
    "bundle literal": "def ask(request: dict) -> dict:\n    return open('cpg/.ccg-work/bundles/x').read()\n",
}


@pytest.mark.parametrize("case", sorted(SERVER_PLANTS))
def test_guard_mcp_tool_surface_fails_closed(tmp_path, case):
    srv = tmp_path / "server.py"
    srv.write_text("from mcp.server.fastmcp import FastMCP\nmcp = FastMCP('ccg')\n@mcp.tool()\n" + SERVER_PLANTS[case])
    assert _escapes(_mcp_tool_patterns(srv)), f"{case} was not flagged"


def test_guard_mcp_tool_with_only_safe_parameters_is_not_flagged(tmp_path):
    srv = tmp_path / "server.py"
    srv.write_text("from mcp.server.fastmcp import FastMCP\nmcp = FastMCP('ccg')\n"
                   "@mcp.tool()\ndef ask(request: dict) -> dict:\n    return {}\n"
                   "@mcp.tool()\ndef refresh(system: str, scope: str = 'all') -> dict:\n    return {}\n"
                   "@mcp.tool()\ndef raw_query(cypher: str) -> dict:\n    return {}\n")
    assert _escapes(_mcp_tool_patterns(srv)) == []


MCP_JSON_PLANTS = {
    "abs cpg root": {"args": ["--root", "/home/u/repo/cpg"]},
    "rel cpg": {"args": ["--root", "cpg/"]},
    "keys path": {"args": ["--root", "cpg/.ccg-work/keys/../bundles"]},
    "bundle dir": {"args": ["--bundles", "cpg/.ccg-work/bundles"]},
    "env path": {"env": {"CCG_DATA": "~/x"}},
    "credentials env": {"env": {"CCG_CREDENTIALS_DIR": "/home/u/.config/ccg/keys"}},
    "launcher var": {"command": "bash", "args": ["-c", "exec \"$CLAUDE_PROJECT_DIR/cpg/clients/mcp/run.sh\""]},
}


@pytest.mark.parametrize("case", sorted(MCP_JSON_PLANTS))
def test_guard_every_mcp_json_ccg_value_is_judged(tmp_path, case):
    m = tmp_path / ".mcp.json"
    m.write_text(json.dumps({"mcpServers": {"ccg": {"command": "x", **MCP_JSON_PLANTS[case]}}}))
    pats, _ = _collect(KIT / "settings.json", KIT / "agents.json", KIT / "run-writer.sh", m, tmp_path / "none.py")
    assert [s for s, _ in _escapes(pats)], f"{case} was not flagged"


def test_guard_mcp_json_plain_words_are_not_flagged(tmp_path):
    m = tmp_path / ".mcp.json"
    m.write_text(json.dumps({"mcpServers": {"ccg": {"command": "ccg-mcp", "args": ["--stdio"]}}}))
    pats, _ = _collect(KIT / "settings.json", KIT / "agents.json", KIT / "run-writer.sh", m, tmp_path / "none.py")
    assert _escapes(pats) == []


# =============================================================================================

def _header():
    """A minimal valid header (the shape of `tests/test_bundle.py`'s `header()`)."""
    import hashlib
    from ccg import bundle, ids
    return {
        "type": "header", "bundleSchemaVersion": "1.0",
        "unit": {"repoId": "graphmind-ai-lab", "rootPath": "salesperson", "lang": "javascript"},
        "unitId": ids.unit_id("graphmind-ai-lab", "salesperson", "javascript"),
        "builtCommit": "1" * 40, "builtTree": "2" * 40, "builtAt": "2026-10-29T12:00:00Z",
        "tools": {"joern": "4.0.579", "frontend": "jssrc2cpg", "frontendVersion": "4.0.579",
                  "normalizer": "normalize.py", "normalizerVersion": "0.0.1", "extractScript": "extract.sc",
                  "extractScriptHash": hashlib.sha256(b"x").hexdigest()},
        "sandboxAuditIds": ["0f8fad5b-d9cb-469f-a165-70867728950e"], "egressMode": "offline",
        "flowProjected": True, "fieldDepthCap": 3, "vendoredDirs": [], "history": "available",
        "counts": {t: 0 for t in bundle.RECORD_TYPES[1:]},
    }


# (g) sweep of interrupted writes (P1-20)
# =============================================================================================

NOW = 2_000_000_000.0
H8, H16 = "0123abcd", "0123456789abcdef"


def _tmp(d: Path, name: str, age_s: float, data: bytes = b"partial") -> Path:
    p = d / name
    p.write_bytes(data)
    p.chmod(0o600)
    os.utime(p, (NOW - age_s, NOW - age_s))
    return p


def _sweep_dir(work) -> Path:
    return atrest.bundle_dir("shop", work)


def test_sweep_removes_old_temp_files_of_both_writers_and_returns_them(work):
    d = _sweep_dir(work)
    a = _tmp(d, f".u1.b1.bundle.jsonl.{H16}.tmp", 4000)          # ccg.bundle.write_bundle's name (16 hex)
    b = _tmp(d, f".u2.b1.bundle.jsonl.{H8}.tmp", 4000)           # the name an older write_private left (8 hex)
    assert atrest.sweep_stale_tmp("shop", root=work, now=NOW) == [a, b]
    assert not a.exists() and not b.exists()


def test_prune_orphans_removes_only_completed_bundles_of_units_not_in_the_live_set(work):
    d = _sweep_dir(work)
    live = _tmp(d, "u1.b1.bundle.jsonl", 0)
    gone = [_tmp(d, "u2.b1.bundle.jsonl", 0), _tmp(d, "u2.b2.bundle.jsonl", 0)]
    tmp = _tmp(d, f".u3.b1.bundle.jsonl.{H16}.tmp", 0)            # an in-progress write is never touched
    other = _tmp(d, "notes.txt", 0)
    odd = _tmp(d, "u4.bundle.jsonl", 0)                           # no <build>
    assert atrest.prune_orphans("shop", {"u1"}, work) == gone
    assert sorted(p.name for p in d.iterdir()) == sorted(p.name for p in (live, tmp, other, odd))


def test_prune_orphans_refuses_a_link_instead_of_following_it(work, tmp_path):
    d = _sweep_dir(work)
    target = tmp_path / "victim"
    target.write_text("keep")
    (d / "u9.b1.bundle.jsonl").symlink_to(target)
    with pytest.raises(atrest.AtRestError):
        atrest.prune_orphans("shop", set(), work)
    assert target.read_text() == "keep"


def _crash_after_create(monkeypatch, call):
    """Run ``call`` with ``os.replace`` failing after the temp file exists and every writer-side cleanup suppressed: a crash's leftover."""
    with monkeypatch.context() as m:
        m.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("crash")))
        m.setattr(atrest, "_unlink_quiet", lambda *a, **k: None)
        m.setattr(Path, "unlink", lambda self, missing_ok=False: None)
        with pytest.raises(OSError, match="crash"):
            call()


@pytest.mark.parametrize("writer", ["atrest.write_private", "atrest.write_bundle", "bundle.write_bundle"])
def test_sweep_removes_the_leftover_of_each_real_writers_crash_by_the_name_that_writer_produced(work, monkeypatch, writer):
    """The names come from the writers themselves (never planted): a writer that changes its temp name fails here, not in production."""
    from ccg import bundle
    d = _sweep_dir(work)
    unit = _header()["unitId"]
    if writer == "atrest.write_private":
        call = lambda: atrest.write_private(atrest.bundle_path("shop", "u1", "b1", work), b"x", atrest._work_root(work))
    elif writer == "atrest.write_bundle":
        call = lambda: atrest.write_bundle("shop", unit, "b1", [_header()], work)
    else:
        call = lambda: bundle.write_bundle(d / f"{unit}.b1.bundle.jsonl", [_header()])
    _crash_after_create(monkeypatch, call)
    left = [p for p in d.iterdir()]
    assert len(left) == 1 and left[0].name.endswith(".tmp") and left[0].name.startswith(".")
    os.utime(left[0], (NOW - 4000, NOW - 4000))
    assert atrest.sweep_stale_tmp("shop", root=work, now=NOW) == left
    assert list(d.iterdir()) == []


def test_write_private_draws_sixteen_hex_digits_like_bundle_write_bundle(work, monkeypatch):
    seen = []
    real = atrest._write_new
    monkeypatch.setattr(atrest, "_write_new", lambda p, data: (seen.append(p.name), real(p, data))[1])
    atrest.write_private(atrest.bundle_path("shop", "u1", "b1", work), b"x", atrest._work_root(work))
    assert re.fullmatch(r"\.u1\.b1\.bundle\.jsonl\.[0-9a-f]{16}\.tmp", seen[0])


@pytest.mark.parametrize("age,gone", [(3601, True), (3600, True), (3599, False), (0, False)])
def test_sweep_age_threshold_is_older_than_s(work, age, gone):
    p = _tmp(_sweep_dir(work), f".u1.b1.bundle.jsonl.{H16}.tmp", age)
    atrest.sweep_stale_tmp("shop", older_than_s=3600, root=work, now=NOW)
    assert p.exists() is not gone


def test_sweep_threshold_is_the_argument_not_a_constant(work):
    p = _tmp(_sweep_dir(work), f".u1.b1.bundle.jsonl.{H16}.tmp", 100)
    atrest.sweep_stale_tmp("shop", root=work, now=NOW)            # default 3600: a 100 s old file stays
    assert p.exists()
    atrest.sweep_stale_tmp("shop", older_than_s=50, root=work, now=NOW)
    assert not p.exists()


@pytest.mark.parametrize("bad", [-1, True, "3600", None])
def test_sweep_refuses_a_malformed_threshold(work, bad):
    with pytest.raises(ValueError):
        atrest.sweep_stale_tmp("shop", older_than_s=bad, root=work, now=NOW)


HOSTILE = [
    f".u1.b1.bundle.jsonl.{H16.upper()}.tmp",                    # upper-case hex
    f".u1.b1.bundle.jsonl.{H16}0.tmp",                           # 17 hex
    f".u1.b1.bundle.jsonl.{H16[:12]}.tmp",                       # 12 hex
    f".u1.b1.bundle.jsonl.{H16}.tmp.keep",                       # a longer name
    f"u1.b1.bundle.jsonl.{H16}.tmp",                             # not hidden
    f".u1.b1.bundle.jsonl.{H16}.tmp\n",                          # trailing newline
    f".u1.b1.bundle.json.{H16}.tmp",                             # not a bundle name
    f".u1.bundle.jsonl.{H16}.tmp",                               # no <build>
    f".u 1.b1.bundle.jsonl.{H16}.tmp",                           # a space in the unit
    f".u1.b1.bundle.jsonl.{H16}.tmp ",
    f"..u1.b1.bundle.jsonl.{H16}.tmp",
    f".{'u' * 65}.b1.bundle.jsonl.{H16}.tmp",                    # a unit longer than the name rule
    f".u1.b1.bundle.jsonl.{H16}.tmp‮",
    ".u1.b1.bundle.jsonl.tmp",
    f".{H16}.tmp",
    "u1.b1.bundle.jsonl",                                        # a completed bundle
    ".sweep-me.tmp",
]


@pytest.mark.parametrize("name", HOSTILE)
def test_sweep_leaves_every_other_name_alone(work, name):
    p = _tmp(_sweep_dir(work), name, 10**6)
    assert atrest.sweep_stale_tmp("shop", root=work, now=NOW) == []
    assert p.exists()


def test_sweep_leaves_a_completed_bundle_and_an_in_progress_one(work):
    done = atrest.write_bundle("shop", "u1", "b1", [_header()], root=work)
    os.utime(done, (NOW - 10**6, NOW - 10**6))
    young = _tmp(_sweep_dir(work), f".u1.b2.bundle.jsonl.{H16}.tmp", 5)
    assert atrest.sweep_stale_tmp("shop", root=work, now=NOW) == []
    assert done.exists() and young.exists()


def test_sweep_touches_only_this_systems_bundle_directory(work):
    mine = _tmp(_sweep_dir(work), f".u1.b1.bundle.jsonl.{H16}.tmp", 10**6)
    other = _tmp(atrest.bundle_dir("other", work), f".u1.b1.bundle.jsonl.{H16}.tmp", 10**6)
    sub = _sweep_dir(work) / "sub"
    sub.mkdir(mode=0o700)
    nested = _tmp(sub, f".u1.b1.bundle.jsonl.{H16}.tmp", 10**6)
    top = _tmp(work, f".u1.b1.bundle.jsonl.{H16}.tmp", 10**6)
    elsewhere = _tmp(work / "bundles", f".u1.b1.bundle.jsonl.{H16}.tmp", 10**6)
    assert atrest.sweep_stale_tmp("shop", root=work, now=NOW) == [mine]
    assert other.exists() and nested.exists() and top.exists() and elsewhere.exists()


def test_sweep_does_not_remove_a_link_a_directory_or_follow_either(work, tmp_path):
    d = _sweep_dir(work)
    victim = tmp_path / "victim.txt"
    victim.write_text("keep")
    link = d / f".u1.b1.bundle.jsonl.{H16}.tmp"
    link.symlink_to(victim)
    os.utime(link, (NOW - 10**6, NOW - 10**6), follow_symlinks=False)
    adir = d / f".u2.b1.bundle.jsonl.{H16}.tmp"
    adir.mkdir(mode=0o700)
    (adir / "inner").write_text("keep")
    assert atrest.sweep_stale_tmp("shop", root=work, now=NOW) == []
    assert link.is_symlink() and victim.read_text() == "keep" and (adir / "inner").exists()


def test_sweep_refuses_a_bundle_directory_that_is_a_link(work, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    bait = _tmp(outside, f".u1.b1.bundle.jsonl.{H16}.tmp", 10**6)
    (work / "bundles").mkdir(mode=0o700)
    (work / "bundles" / "shop").symlink_to(outside)
    with pytest.raises(AtRestError, match="symlink"):
        atrest.sweep_stale_tmp("shop", root=work, now=NOW)
    assert bait.exists()


def test_sweep_refuses_a_bundles_directory_that_is_a_link(work, tmp_path):
    outside = tmp_path / "outside"
    (outside / "shop").mkdir(parents=True, mode=0o700)
    bait = _tmp(outside / "shop", f".u1.b1.bundle.jsonl.{H16}.tmp", 10**6)
    (work / "bundles").symlink_to(outside)
    with pytest.raises(AtRestError, match="symlink"):
        atrest.sweep_stale_tmp("shop", root=work, now=NOW)
    assert bait.exists()


def test_sweep_refuses_a_system_name_that_is_a_path(work):
    for bad in ("../shop", "shop/x", "", "a\0b"):
        with pytest.raises(AtRestError):
            atrest.sweep_stale_tmp(bad, root=work, now=NOW)


def test_sweep_of_an_absent_directory_removes_nothing(work):
    assert atrest.sweep_stale_tmp("fresh", root=work, now=NOW) == []


def test_sweep_opens_the_directory_without_following_a_link_swapped_in_after_the_name_check(work, tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    bait = _tmp(outside, f".u1.b1.bundle.jsonl.{H16}.tmp", 10**6)
    link = tmp_path / "swapped"
    link.symlink_to(outside)
    monkeypatch.setattr(atrest, "bundle_dir", lambda system, root=None: link)       # what a swap between the check and the open looks like
    with pytest.raises(AtRestError, match="cannot open"):
        atrest.sweep_stale_tmp("shop", root=work, now=NOW)
    assert bait.exists()


def test_sweep_unlinks_relative_to_the_directory_it_opened(work, monkeypatch):
    p = _tmp(_sweep_dir(work), f".u1.b1.bundle.jsonl.{H16}.tmp", 10**6)
    calls = []
    real = os.unlink
    monkeypatch.setattr(os, "unlink", lambda path, *a, **k: (calls.append(k.get("dir_fd")), real(path, *a, **k))[1])
    atrest.sweep_stale_tmp("shop", root=work, now=NOW)
    assert calls and all(fd is not None for fd in calls) and not p.exists()
