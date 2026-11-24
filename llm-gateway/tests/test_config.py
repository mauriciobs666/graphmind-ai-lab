"""Configuration parser and owner trust (plan §3.5, §3.6; AC-4, AC-21)."""

import os

import pytest

from llmgateway import config as config_mod
from llmgateway.config import (
    DEFAULT_PATH,
    ENV_VAR,
    ConfigError,
    ConfigSource,
    parse_origin,
    resolve_path,
)

ME = os.getuid()

TOML = """
[deployment]
local_endpoints = ["http://127.0.0.1:1234", "http://[::1]:11434", "http://10.1.2.3:8080",
                   "http://localhost:1", "http://user@127.0.0.1:9", "not a url"]
holder_socket = "/run/llmgateway/holder.sock"

[holder]
unmatched = "refuse"

[providers.lm]
kind = "lmstudio"
base_url = "http://127.0.0.1:1234/v1"

[providers.remote]
kind = "openai_compat"
base_url = "https://api.example.com/v1"

[providers.claude]
kind = "anthropic"
base_url = "https://api.anthropic.com"

[scopes.alpha]
marker = "restricted"

[scopes."ccg:system:main"]
record_text = true

[scope_sources.ccg-manifest]
prefix = "ccg:"

[call_sites.chat]
component = "x"
scopes = ["alpha"]
chain = ["lm", "claude", { provider = "remote" }]
local_chain = ["lm"]

[[scope_overrides]]
scope = "alpha"
chain = ["claude"]

[[run_bounds]]
match = "eval-*"
max_tokens = 100
"""


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "root"
    (r / "etc").mkdir(parents=True)
    r.chmod(0o755)
    (r / "etc").chmod(0o755)
    return r


def write(root, text=TOML, mode=0o644):
    f = root / "etc" / "gateway.toml"
    f.write_text(text)
    f.chmod(mode)
    return f


def source(root, trusted=frozenset({ME})):
    return ConfigSource(root / "etc" / "gateway.toml", trusted, anchor=root)


# --- path resolution ----------------------------------------------------------------------


def test_env_overrides_path_in_library_only():
    env = {ENV_VAR: "/somewhere/else.toml"}
    assert resolve_path(env, library=True) == "/somewhere/else.toml"
    assert resolve_path(env, library=False) == DEFAULT_PATH
    assert resolve_path(env) == DEFAULT_PATH
    assert resolve_path({}, library=True) == DEFAULT_PATH
    assert resolve_path({ENV_VAR: ""}, library=True) == DEFAULT_PATH


def test_library_reads_process_env_by_default(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "/x.toml")
    assert resolve_path(library=True) == "/x.toml"
    assert resolve_path() == DEFAULT_PATH


def test_loading_never_reads_env(root, monkeypatch):
    other = root / "etc" / "other.toml"
    other.write_text('[deployment]\nlocal_endpoints = ["http://127.0.0.1:1"]\n')
    other.chmod(0o644)
    write(root)
    monkeypatch.setenv(ENV_VAR, str(other))
    assert source(root).get().is_local("http://127.0.0.1:1234")
    assert not source(root).get().is_local("http://127.0.0.1:1")


# --- protected configuration --------------------------------------------------------------


def test_protected_config_is_used(root):
    write(root)
    cfg = source(root).get()
    assert cfg.protected and cfg.hosted_enabled and not cfg.loopback_only
    assert set(cfg.scopes) == {"alpha", "ccg:system:main"}
    assert cfg.scope_overrides[0]["scope"] == "alpha"
    assert cfg.run_bounds[0]["match"] == "eval-*"
    assert set(cfg.providers) == {"lm", "remote", "claude"}
    assert cfg.call_sites["chat"]["scopes"] == ["alpha"]
    assert cfg.holder["unmatched"] == "refuse"


def test_effective_local_endpoints_are_canonical_origins_only(root):
    write(root)
    cfg = source(root).get()
    assert cfg.effective_local_endpoints() == {
        ("http", "127.0.0.1", 1234),
        ("http", "::1", 11434),
        ("http", "10.1.2.3", 8080),
    }
    assert len(cfg.warnings) == 3  # hostname, userinfo, garbage


def test_is_local_matches_listed_origins_only(root):
    write(root)
    cfg = source(root).get()
    assert cfg.is_local("http://127.0.0.1:1234/v1/chat")
    assert cfg.is_local("http://[::1]:11434")
    assert cfg.is_local("http://10.1.2.3:8080")
    assert not cfg.is_local("http://127.0.0.1:1235")  # port differs
    assert not cfg.is_local("https://127.0.0.1:1234")  # scheme differs
    assert not cfg.is_local("http://127.0.0.2:1234")  # not listed, though loopback
    assert not cfg.is_local("https://api.example.com/v1")


def test_provider_hosted_classification(root):
    write(root)
    cfg = source(root).get()
    assert not cfg.provider_is_hosted("lm")
    assert cfg.provider_is_hosted("remote")  # AC-4: a remote OpenAI-compatible URL is hosted
    assert cfg.provider_is_hosted("claude")


def test_static_marker_under_scope_source_prefix_is_rejected(root):
    write(root, TOML + '\n[scopes."ccg:repo:x"]\nmarker = "unrestricted"\n')
    with pytest.raises(ConfigError, match="source-backed"):
        source(root).get()


@pytest.mark.parametrize(
    "scopes",
    [
        '[scopes."Workspace:acme"]\nmarker = "restricted"\n',
        '[scopes."a b"]\nmarker = "restricted"\n',
        '[scopes."аlpha"]\nmarker = "restricted"\n',  # Cyrillic a
        f'[scopes."{"a" * 129}"]\nmarker = "restricted"\n',
        '[scopes.ok]\nmarker = "loose"\n',
    ],
)
def test_ill_formed_scope_or_marker_is_rejected(root, scopes):
    write(root, scopes)
    with pytest.raises(ConfigError):
        source(root).get()


def test_scope_id_of_128_chars_is_accepted(root):
    write(root, f'[scopes."{"a" * 128}"]\nmarker = "restricted"\n')
    assert source(root).get().protected


def test_invalid_toml_in_protected_file_is_a_config_error(root):
    write(root, "this is = = not toml")
    with pytest.raises(ConfigError, match="TOML"):
        source(root).get()


# --- unprotected: fail closed (§3.5) ------------------------------------------------------


@pytest.mark.parametrize("how", ["group_writable", "wrong_owner", "writable_ancestor"])
def test_failing_owner_check_collapses_to_loopback_and_ignores_owner_sections(root, how):
    f = write(root)
    trusted = frozenset({ME})
    if how == "group_writable":
        f.chmod(0o664)
    elif how == "wrong_owner":
        trusted = frozenset({ME + 4242})
    else:
        (root / "etc").chmod(0o775)
    cfg = source(root, trusted).get()
    assert not cfg.protected and not cfg.hosted_enabled and cfg.loopback_only
    assert cfg.reason
    # local_endpoints ignored: the listed 10.1.2.3 and ::1:11434 do not count ...
    assert cfg.effective_local_endpoints() == frozenset()
    assert not cfg.is_local("http://10.1.2.3:8080")
    # ... the allow-list is loopback IP literals, any port.
    assert cfg.is_local("http://127.0.0.1:9999")
    assert cfg.is_local("https://[::1]:1")
    assert cfg.is_local("http://127.5.5.5:80")
    assert not cfg.is_local("http://localhost:1234")
    assert not cfg.is_local("https://api.example.com")
    # scopes, overrides, run bounds, deployment, holder: ignored.
    assert cfg.scopes == {} and cfg.scope_sources == {}
    assert cfg.scope_overrides == () and cfg.run_bounds == ()
    assert cfg.deployment == {} and cfg.holder == {}
    # hosted steps: hosted providers and every chain step naming one are gone.
    assert set(cfg.providers) == {"lm"}
    assert cfg.call_sites["chat"]["chain"] == ["lm"]
    assert cfg.call_sites["chat"]["local_chain"] == ["lm"]
    assert cfg.call_sites["chat"]["scopes"] == []


def test_unprotected_remote_url_listed_as_local_is_not_local(root):
    write(
        root,
        '[deployment]\nlocal_endpoints = ["https://remote.example:443"]\n'
        '[providers.p]\nkind = "openai_compat"\nbase_url = "https://remote.example/v1"\n',
        mode=0o666,
    )
    cfg = source(root).get()
    assert not cfg.is_local("https://remote.example/v1")
    assert cfg.providers == {}


def test_unprotected_with_garbage_or_missing_file_is_empty_loopback_only(root):
    write(root, "= = garbage", mode=0o666)
    cfg = source(root).get()
    assert not cfg.protected and cfg.providers == {} and cfg.is_local("http://127.0.0.1:1")
    (root / "etc" / "gateway.toml").unlink()
    cfg = source(root).get()
    assert not cfg.protected and "exist" in cfg.reason


def test_symlinked_config_yields_nothing_at_all(root):
    real = root / "etc" / "real.toml"
    real.write_text(TOML)
    real.chmod(0o644)
    link = root / "etc" / "gateway.toml"
    link.symlink_to(real)
    cfg = source(root).get()
    assert not cfg.protected and cfg.providers == {} and cfg.call_sites == {}


def test_recheck_on_every_call_tightening_and_loosening(root):
    f = write(root)
    src = source(root)
    assert src.get().protected
    f.chmod(0o666)
    assert not src.get().protected
    f.chmod(0o644)
    assert src.get().protected
    (root / "etc").chmod(0o777)
    assert not src.get().protected


def test_parse_is_cached_by_stat_key_but_check_is_not(root, monkeypatch):
    f = write(root)
    calls = []
    real = config_mod.tomllib.loads
    monkeypatch.setattr(config_mod.tomllib, "loads", lambda s: calls.append(1) or real(s))
    src = source(root)
    first = src.get()
    assert src.get() is first and len(calls) == 1
    f.chmod(0o666)  # check re-runs even though the parse is cached
    assert not src.get().protected
    f.chmod(0o644)
    src.get()
    f.write_text(TOML.replace("1234", "4321"))  # size same, mtime changes
    os.utime(f, ns=(1, 2_000_000_000))
    assert src.get().is_local("http://127.0.0.1:4321")


# --- is_local / canonical origin ----------------------------------------------------------


@pytest.mark.parametrize(
    "url,origin",
    [
        ("http://127.0.0.1", ("http", "127.0.0.1", 80)),
        ("https://127.0.0.1", ("https", "127.0.0.1", 443)),
        ("HTTP://127.0.0.1:1234/v1?x=1#f", ("http", "127.0.0.1", 1234)),
        ("http://[::1]:5", ("http", "::1", 5)),
        ("http://[0:0:0:0:0:0:0:1]:5", ("http", "::1", 5)),
        ("http://[::1]", ("http", "::1", 80)),
    ],
)
def test_canonical_origin(url, origin):
    assert parse_origin(url) == origin


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:1234",
        "http://example.com",
        "http://user@127.0.0.1:1234",
        "http://user:pw@127.0.0.1",
        "http://127.0.0.1:1234@evil.com",
        "http://127.1",
        "http://0x7f.0.0.1",
        "http://2130706433",
        "http://127.000.000.001",
        "http://[::1%25lo]:1",
        "http://[::1%lo]:1",
        "http://[::1",
        "http://[::1]x:1",
        "http://127.0.0.1:",
        "http://127.0.0.1:0",
        "http://127.0.0.1:65536",
        "http://127.0.0.1:+80",
        "http://127.0.0.1:١٢",
        "http://127.0.0.1 :80",
        "http://127.0.0.1\\@evil",
        "http://127.0.0.1\n",
        "ftp://127.0.0.1",
        "//127.0.0.1",
        "127.0.0.1:1234",
        "http:///x",
        "",
        None,
        7,
    ],
)
def test_malformed_or_non_literal_origins_are_refused(url):
    assert parse_origin(url) is None
    cfg = config_mod.GatewayConfig(protected=False)
    assert not cfg.is_local(url)


@pytest.mark.parametrize(
    "url", ["http://[::ffff:127.0.0.1]:1", "http://0.0.0.0:1", "http://10.0.0.1:1"]
)
def test_unprotected_loopback_is_literal_loopback_only(url):
    assert not config_mod.GatewayConfig(protected=False).is_local(url)


def test_unprotected_never_honours_owner_origins_even_if_constructed_with_them():
    cfg = config_mod.GatewayConfig(
        protected=False, local_origins=frozenset({("http", "10.1.2.3", 8080)})
    )
    assert cfg.effective_local_endpoints() == frozenset()
    assert not cfg.is_local("http://10.1.2.3:8080")
    assert cfg.is_local("http://127.0.0.1:8080")
