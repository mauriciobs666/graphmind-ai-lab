"""D9b — the broker's start-up, in process (egress-spec5 section 1 and section 7 items 8 and 9): `load_manifest_text`, the
static check of hosted entries, owner-protected manifest loading (parsed from the verified bytes, once), the option rules
of `main`, and the credential intake. The real-process variants are in `test_broker_startup_process.py`.

`trusted_uids` is `{0, own uid}` here (the real `/etc/llmgateway` chain is root-owned: tier 2). Directories are under the
user's cache dir, not `/tmp`: the broker checks every ancestor strictly and `/tmp` is sticky world-writable."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from broker_helpers import HOST_ADDR, egress, fake_iface, hosted_egress, write_manifest
from ccg import manifest as mf
from ccg.broker import ownerprot, server
from ccg.broker.audit import PreconditionError
from ccg.broker.holderclient import HolderClient

ME = os.getuid()
TRUSTED = frozenset({0, ME})
SECRET_HEX = "0123456789abcdef" * 4


@pytest.fixture
def tmp_path():
    base = Path.home() / ".cache" / "ccg-broker-tests"
    base.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(dir=base))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def hosted_at(endpoint: str, **entry) -> dict:
    return hosted_egress(endpoint, **entry)


def load(tmp_path, eg, **kw):
    return server.load_systems([str(write_manifest(tmp_path, eg=eg))], fake_iface(), **kw)


# ------------------------------------------------------------------ load_manifest_text


def test_load_manifest_text_equals_load_manifest_of_the_same_bytes(tmp_path):
    p = write_manifest(tmp_path)
    assert mf.load_manifest_text(p.read_text()) == mf.load_manifest(p)


def test_load_manifest_text_refuses_what_load_manifest_refuses(tmp_path):
    with pytest.raises(mf.ManifestError, match="not valid YAML"):
        mf.load_manifest_text("a: [")
    with pytest.raises(mf.ManifestError, match="YAML mapping"):
        mf.load_manifest_text("- 1\n")
    with pytest.raises(mf.ManifestError, match="duplicate key"):
        mf.load_manifest_text("a: 1\na: 2\n")
    bad = tmp_path / "bad.yaml"
    bad.write_text("schemaVersion: 1\n")
    with pytest.raises(mf.ManifestError) as a:
        mf.load_manifest(bad)
    with pytest.raises(mf.ManifestError) as b:
        mf.load_manifest_text(bad.read_text())
    assert str(a.value) == str(b.value)


def test_load_manifest_still_reads_by_path_and_raises_oserror_for_a_missing_file(tmp_path):
    with pytest.raises(OSError):
        mf.load_manifest(tmp_path / "nope.yaml")


# ------------------------------------------------------------------ the static check of hosted entries


@pytest.mark.parametrize("endpoint", [
    "https://127.0.0.1:443", "https://127.1.2.3:8443", "https://[::1]:443", "https://10.1.2.3:443",
    "https://192.168.5.5:443", "https://172.20.0.9:443", "https://100.64.0.1:443", "https://[fc00::1]:443",
    f"https://{HOST_ADDR}:443", "https://192.168.0.1:443", "https://172.17.0.5:443", "https://169.254.1.1:443",
    "https://0.0.0.0:443", "https://224.0.0.1:443", "https://[fe80::1]:443",
])
def test_a_forbidden_ip_literal_on_a_hosted_entry_fails_at_load_naming_the_file(tmp_path, endpoint):
    p = write_manifest(tmp_path, eg=hosted_at(endpoint))
    with pytest.raises(mf.ManifestError) as e:
        server.load_systems([str(p)], fake_iface())
    assert str(p) in str(e.value) and endpoint in str(e.value) and "forbidden address literal" in str(e.value)


def test_the_private_address_flag_does_not_admit_a_private_literal_on_a_hosted_entry(tmp_path):
    with pytest.raises(mf.ManifestError, match="forbidden address literal"):
        load(tmp_path, hosted_at("https://10.1.2.3:443", privateAddress=True))
    assert load(tmp_path, hosted_at("https://93.184.216.34:443", privateAddress=True))  # a public one still loads


@pytest.mark.parametrize("endpoint", ["https://93.184.216.34:443", "https://api.provider.example.com:443",
                                      "https://[2606:2800:220:1:248:1893:25c8:1946]:443"])
def test_a_public_literal_and_a_name_host_pass_the_static_check(tmp_path, endpoint):
    assert list(load(tmp_path, hosted_at(endpoint))) == ["shop"]


def test_the_check_off_seam_admits_loopback_hosted_fixtures_and_is_the_only_way(tmp_path):
    eg = hosted_at("https://127.0.0.1:8443")
    assert list(load(tmp_path, eg, check_hosted_entries=False)) == ["shop"]
    with pytest.raises(mf.ManifestError):
        load(tmp_path, eg)  # on by default
    import inspect
    assert inspect.signature(server.load_systems).parameters["check_hosted_entries"].default is True
    assert inspect.signature(server.main).parameters["check_hosted_entries"].default is True
    src = Path(server.__file__).read_text()
    assert "environ.get(\"CCG_CHECK" not in src and "--check-hosted" not in src  # no option, no variable


def test_a_hosted_scheme_other_than_https_is_refused_by_the_static_check_itself():
    eg = {"local": [], "hosted": [{"id": "h", "endpoint": "http://api.example.com:80"}]}
    bad = server._check_hosted_entries(eg, fake_iface())
    assert bad == ["hosted endpoint http://api.example.com:80 (h) must be https"]
    # and the manifest schema refuses it before that, so a manifest never reaches the check with one
    with pytest.raises(mf.ManifestError):
        mf.load_manifest_text(yaml.safe_dump({"schemaVersion": 1, "system": "shop", "egress": {
            **egress(), "hosted": [{"id": "h", "kind": "model", "endpoint": "http://api.example.com:80",
                                    "paths": ["/x"]}]}, "repos": [{"repoId": "a", "path": ".", "policy": {"egress": "hosted"}}]}))


def test_the_interface_data_is_read_only_for_a_hosted_ip_literal(tmp_path):
    calls = []

    def iface():
        calls.append(1)
        return fake_iface()()

    p = write_manifest(tmp_path, eg=hosted_at("https://api.provider.example.com:443"))
    server.load_systems([str(p)], iface)
    assert calls == []  # a name host, a loopback local entry: nothing to read
    p2 = write_manifest(tmp_path, eg=hosted_at("https://93.184.216.34:443"), name="b.yaml")
    server.load_systems([str(p2)], iface)
    assert calls == [1]


def test_an_unreadable_interface_table_stops_the_load_for_a_hosted_literal(tmp_path):
    from ccg.broker.addr import IfaceUnavailable

    def broken():
        raise IfaceUnavailable("no ip")

    p = write_manifest(tmp_path, eg=hosted_at("https://93.184.216.34:443"))
    with pytest.raises(mf.ManifestError, match="interface data"):
        server.load_systems([str(p)], broken)


# ------------------------------------------------------------------ owner-protected loading (item 8)


def protected(tmp_path, **kw):
    return server.load_systems([str(write_manifest(tmp_path))], fake_iface(), require_owner_protected=True,
                               trusted_uids=kw.pop("trusted_uids", TRUSTED), **kw)


def test_a_conforming_manifest_loads_when_owner_protection_is_required(tmp_path):
    assert list(protected(tmp_path)) == ["shop"]


@pytest.mark.parametrize("mode", [0o664, 0o646, 0o666])
def test_a_user_group_or_other_writable_manifest_is_refused(tmp_path, mode):
    p = write_manifest(tmp_path)
    p.chmod(mode)
    with pytest.raises(PreconditionError) as e:
        server.load_systems([str(p)], fake_iface(), require_owner_protected=True, trusted_uids=TRUSTED)
    assert str(p) in str(e.value) and "group- or other-writable" in str(e.value)


def test_a_manifest_owned_by_an_untrusted_uid_is_refused(tmp_path):
    p = write_manifest(tmp_path)
    for trusted in (frozenset({0}), frozenset({ME + 1})):  # own uid not trusted (production `{0}`), or another one
        with pytest.raises(PreconditionError) as e:
            server.load_systems([str(p)], fake_iface(), require_owner_protected=True, trusted_uids=trusted)
        assert str(p) in str(e.value) and "untrusted uid" in str(e.value)


def test_a_symlinked_manifest_and_a_symlinked_ancestor_are_refused(tmp_path):
    p = write_manifest(tmp_path)
    link = tmp_path / "link.yaml"
    link.symlink_to(p)
    with pytest.raises(PreconditionError, match="symlink"):
        server.load_systems([str(link)], fake_iface(), require_owner_protected=True, trusted_uids=TRUSTED)
    real = tmp_path / "real"
    real.mkdir()
    write_manifest(real)
    (tmp_path / "via").symlink_to(real)
    with pytest.raises(PreconditionError, match="symlink"):
        server.load_systems([str(tmp_path / "via" / "system.yaml")], fake_iface(), require_owner_protected=True,
                            trusted_uids=TRUSTED)


@pytest.mark.parametrize("mode", [0o775, 0o757, 0o777])
def test_a_manifest_under_a_group_or_other_writable_ancestor_is_refused(tmp_path, mode):
    d = tmp_path / "etc"
    d.mkdir()
    p = write_manifest(d)
    d.chmod(mode)
    with pytest.raises(PreconditionError, match="ancestor 'etc' is group- or other-writable"):
        server.load_systems([str(p)], fake_iface(), require_owner_protected=True, trusted_uids=TRUSTED)


def test_a_fifo_and_a_missing_manifest_are_refused_without_hanging(tmp_path):
    fifo = tmp_path / "f.yaml"
    os.mkfifo(fifo, 0o644)
    with pytest.raises(PreconditionError, match="not a regular file"):
        server.load_systems([str(fifo)], fake_iface(), require_owner_protected=True, trusted_uids=TRUSTED)
    with pytest.raises(PreconditionError, match="does not exist"):
        server.load_systems([str(tmp_path / "nope.yaml")], fake_iface(), require_owner_protected=True,
                            trusted_uids=TRUSTED)


def test_without_the_flag_a_user_writable_manifest_still_loads(tmp_path):
    p = write_manifest(tmp_path)
    p.chmod(0o666)
    assert list(server.load_systems([str(p)], fake_iface())) == ["shop"]  # the flag is what enforces ownership


def test_a_manifest_swapped_between_the_check_and_the_parse_is_not_loaded(tmp_path, monkeypatch):
    p = write_manifest(tmp_path, system="shop")
    evil = write_manifest(tmp_path, system="evil", name="evil.yaml")
    real = ownerprot.require_owner_file

    def swapping(path, trusted, **kw):
        data = real(path, trusted, **kw)
        os.replace(evil, path)  # the gap: after the verified read, before the parse
        return data

    def never(*a, **kw):
        raise AssertionError("the manifest was re-read by path")

    monkeypatch.setattr(ownerprot, "require_owner_file", swapping)
    monkeypatch.setattr(mf, "load_manifest", never)
    systems = server.load_systems([str(p)], fake_iface(), require_owner_protected=True, trusted_uids=TRUSTED)
    assert list(systems) == ["shop"] and "evil" in p.read_text()


def test_a_manifest_that_is_not_utf8_is_a_configuration_error_not_a_crash(tmp_path):
    p = tmp_path / "bin.yaml"
    p.write_bytes(b"\xff\xfe\x00bad")
    p.chmod(0o644)
    with pytest.raises(mf.ManifestError, match="not valid UTF-8"):
        server.load_systems([str(p)], fake_iface(), require_owner_protected=True, trusted_uids=TRUSTED)


# ------------------------------------------------------------------ main: option rules (item 8), no side effects


def run_main(tmp_path, monkeypatch, capsys, args, **kw):
    d = tmp_path / "b"
    d.mkdir(exist_ok=True, mode=0o700)
    monkeypatch.setenv("CCG_AUDIT_DIR", str(d / "audit"))
    # a start that passes validation must return, not serve: a regression then fails an assertion instead of hanging
    monkeypatch.setattr(server.BrokerServer, "serve_forever", lambda self: None)
    rc = server.main(args(d) if callable(args) else args, trusted_uids=kw.pop("trusted_uids", TRUSTED), **kw)
    return rc, capsys.readouterr().err, d


def cred_file(tmp_path, text=SECRET_HEX + "\n", mode=0o600) -> Path:
    p = tmp_path / "broker.cred"
    p.write_text(text)
    p.chmod(mode)
    return p


def base_args(tmp_path, *extra):
    def build(d):
        return ["--system", str(write_manifest(d)), "--socket", str(d / "broker.sock"), *extra]
    return build


def no_side_effects(d):
    assert not (d / "broker.sock").exists() and not (d / "broker.lock").exists() and not (d / "audit").exists()


@pytest.mark.parametrize("extra,needle", [
    (["--key-holder-socket", "/x/h.sock", "--holder-uid", str(ME + 1)], "--require-owner-protected"),
    (["--key-holder-socket", "/x/h.sock", "--require-owner-protected"], "--holder-uid"),
    (["--holder-cred", "/x/c"], "--holder-cred is valid only with --key-holder-socket"),
    (["--holder-uid", str(ME + 1)], "--holder-uid is valid only with --key-holder-socket"),
    (["--key-holder-socket", "/x/h.sock", "--require-owner-protected", "--holder-uid", str(ME)], "other than the broker's own"),
    (["--key-holder-socket", "/x/h.sock", "--require-owner-protected", "--holder-uid", "-1"], "other than the broker's own"),
    (["--key-holder-socket", "/x/h.sock", "--require-owner-protected", "--holder-uid", "abc"], "invalid int"),
])
def test_an_option_violation_exits_64_before_anything_is_bound_locked_or_created(tmp_path, monkeypatch, capsys, extra, needle):
    rc, err, d = run_main(tmp_path, monkeypatch, capsys, base_args(tmp_path, *extra))
    assert rc == 64 and needle in err
    no_side_effects(d)


def test_the_holder_uid_equal_to_the_broker_uid_is_refused_whatever_trusted_uids_is(tmp_path, monkeypatch, capsys):
    extra = ["--key-holder-socket", "/x/h.sock", "--require-owner-protected", "--holder-uid", str(ME),
             "--holder-cred", str(cred_file(tmp_path))]
    rc, err, d = run_main(tmp_path, monkeypatch, capsys, base_args(tmp_path, *extra), trusted_uids=frozenset({0}))
    assert rc == 64 and "other than the broker's own" in err
    no_side_effects(d)


def test_a_hosted_http_or_forbidden_literal_entry_exits_64_with_no_side_effects(tmp_path, monkeypatch, capsys):
    for i, endpoint in enumerate(("https://10.0.0.5:443", "https://127.0.0.1:443")):
        def args(d, endpoint=endpoint, i=i):
            m = write_manifest(d, eg=hosted_at(endpoint), name=f"m{i}.yaml")
            return ["--system", str(m), "--socket", str(d / "broker.sock")]
        rc, err, d = run_main(tmp_path, monkeypatch, capsys, args)
        assert rc == 64 and endpoint in err and "forbidden address literal" in err
        no_side_effects(d)


# ------------------------------------------------------------------ the credential intake (item 9)


def holder_args(tmp_path, cred: Path, *extra):
    return base_args(tmp_path, "--require-owner-protected", "--key-holder-socket", str(tmp_path / "h.sock"),
                     "--holder-uid", str(ME + 1), "--holder-cred", str(cred), *extra)


@pytest.mark.parametrize("text", ["", "ab\n", "00" * 31, "00" * 33, "zz" * 32, "00 " * 32, SECRET_HEX + "00", "é" * 64,
                                  "0x" + "00" * 31 + "\n"])
def test_a_malformed_credential_exits_64_and_names_no_byte_of_it(tmp_path, monkeypatch, capsys, text):
    c = cred_file(tmp_path, text)
    rc, err, d = run_main(tmp_path, monkeypatch, capsys, holder_args(tmp_path, c))
    assert rc == 64 and str(c) in err and "exactly 32 bytes" in err
    assert text.strip() == "" or text.strip() not in err
    no_side_effects(d)


def test_an_oversize_credential_is_a_usage_error_not_a_read_of_everything(tmp_path, monkeypatch, capsys):
    c = cred_file(tmp_path, SECRET_HEX + " " * 10000)
    rc, err, d = run_main(tmp_path, monkeypatch, capsys, holder_args(tmp_path, c))
    assert rc == 64


def test_a_missing_symlinked_loose_mode_and_multiply_linked_credential_exits_69(tmp_path, monkeypatch, capsys):
    good = cred_file(tmp_path)
    link = tmp_path / "link.cred"
    link.symlink_to(good)
    twin = tmp_path / "twin.cred"
    os.link(good, twin)  # `good` and `twin` are now one inode with two links
    fifo = tmp_path / "fifo.cred"
    os.mkfifo(fifo, 0o600)
    loose = tmp_path / "loose.cred"
    loose.write_text(SECRET_HEX)
    loose.chmod(0o640)
    cases = {"missing": (tmp_path / "nope.cred", "cannot open"), "symlink": (link, "cannot open"),
             "two links": (twin, "more than one link"), "fifo": (fifo, "not a regular file"),
             "mode 0640": (loose, "accessible to group or other")}
    for name, (path, needle) in cases.items():
        rc, err, d = run_main(tmp_path, monkeypatch, capsys, holder_args(tmp_path, path))
        assert (name, rc) == (name, 69), err
        assert str(path) in err and needle in err, (name, err)
        no_side_effects(d)


def test_a_credential_owned_by_another_uid_is_refused(tmp_path, monkeypatch):
    c = cred_file(tmp_path)
    monkeypatch.setattr(server.os, "getuid", lambda: ME + 7)
    with pytest.raises(PreconditionError, match="not owned by the broker's uid"):
        server.read_credential(str(c))


def test_a_good_credential_is_read_once_and_builds_the_holder_client(tmp_path, monkeypatch, capsys):
    c = cred_file(tmp_path)
    opened, built = [], {}
    real_open = os.open

    def counting(path, *a, **kw):
        if os.fspath(path) == str(c):
            opened.append(1)
        return real_open(path, *a, **kw)

    class FakeBroker(server.Broker):
        def __init__(self, *a, **kw):
            built.update(kw)
            super().__init__(*a, **kw)

    monkeypatch.setattr(server.os, "open", counting)
    monkeypatch.setattr(server, "Broker", FakeBroker)
    rc, err, d = run_main(tmp_path, monkeypatch, capsys, holder_args(tmp_path, c))
    assert rc == 0, err
    hc = built["hosted_exchanger"]
    assert isinstance(hc, HolderClient) and hc.socket_path == str(tmp_path / "h.sock") and hc.holder_uid == ME + 1
    assert hc._secret == bytes.fromhex(SECRET_HEX) and opened == [1]
    assert SECRET_HEX not in repr(hc) and SECRET_HEX not in err
    assert not (tmp_path / "h.sock").exists()  # startup never connects to or requires the holder


def test_the_default_credential_path_is_the_passwd_home_config(monkeypatch):
    monkeypatch.setenv("HOME", "/somewhere/else")
    import pwd
    assert server.default_holder_cred() == f"{pwd.getpwuid(ME).pw_dir}/.config/llmgateway/callers/broker.cred"


def test_without_a_holder_socket_no_credential_is_read_and_no_exchanger_is_built(tmp_path, monkeypatch, capsys):
    built = {}

    class FakeBroker(server.Broker):
        def __init__(self, *a, **kw):
            built.update(kw)
            super().__init__(*a, **kw)

    monkeypatch.setattr(server, "Broker", FakeBroker)
    monkeypatch.setattr(server, "read_credential", lambda p: pytest.fail("the credential was read"))
    rc, err, d = run_main(tmp_path, monkeypatch, capsys, base_args(tmp_path, "--require-owner-protected"))
    assert rc == 0 and built["hosted_exchanger"] is None


def test_reexec_runs_only_in_the_main_guard_and_main_reads_the_credential_after_it():
    src = Path(server.__file__).read_text()
    guard = src.index('if __name__ == "__main__":\n    reexec_if_needed')
    assert src.count("reexec_if_needed(") == 2  # its definition and the one call in the guard
    assert src.index("def main(") > guard and src.index("read_credential(args.holder_cred") > guard
