"""The holder selects a provider by canonical origin (plan §3.4; review P5-1): the ccg broker sends
`scheme://host:port` with an explicit port, an owner writes `base_url` in any spelling."""

import ast
import pathlib

import pytest
from holder_harness import Clock, Env, StandIn, body_for, call, make_cert

from llmgateway.config import GatewayConfig
from llmgateway.keyholder import HolderStartError, build_policy, canonical_origin, https_upstream

CCG_MANIFEST = pathlib.Path(__file__).resolve().parents[2] / "cpg" / "ccg" / "manifest.py"

SPELLINGS = [
    "https://api.anthropic.com",
    "https://api.anthropic.com/",
    "HTTPS://API.Anthropic.COM:443",
    "https://api.anthropic.com:8443/v1",
    "http://127.0.0.1",
    "https://[::1]",
    "https://[0:0:0:0:0:0:0:1]:9",
    "https://[2001:DB8:0:0:0:0:0:1]",
    "https://u:p@host.example:1/x?y#z",
    "https://host.example:65535",
]


def _broker_canonical_endpoint():
    """The broker's `canonical_endpoint`, extracted by symbol from the read-only source (importing
    `ccg.manifest` would need its YAML/jsonschema stack)."""
    tree = ast.parse(CCG_MANIFEST.read_text())
    wanted = {"_bad_char", "canonical_endpoint", "ManifestError"}
    nodes = [n for n in tree.body if getattr(n, "name", None) in wanted]
    nodes += [
        n
        for n in tree.body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "_PORTS"
    ]
    assert {getattr(n, "name", "_PORTS") for n in nodes} == wanted | {"_PORTS"}
    ns = {}
    exec(  # noqa: S102 - our own repo's source, nothing else
        "import ipaddress\nfrom urllib.parse import urlsplit\n"
        + "\n".join(ast.unparse(n) for n in nodes),
        ns,
    )
    return ns["canonical_endpoint"], ns["ManifestError"]


@pytest.mark.parametrize("url", SPELLINGS)
def test_holder_and_broker_canonicalise_identically(url):
    broker, _ = _broker_canonical_endpoint()
    assert canonical_origin(url) == broker(url)


@pytest.mark.parametrize(
    "url",
    ["", "ftp://h", "https://", "https://h:0", "https://h:99999", "https://h\n", "https://h x",
     "https://[::1%25eth0]", "https://[zz]", "https://hé", "https://h:abc"],
)
def test_holder_and_broker_refuse_the_same_urls(url):
    broker, err = _broker_canonical_endpoint()
    with pytest.raises(err):
        broker(url)
    with pytest.raises(ValueError):
        canonical_origin(url)


@pytest.fixture(scope="module")
def cert(tmp_path_factory):
    return make_cert(tmp_path_factory.mktemp("tls"))


@pytest.fixture
def env(tmp_path, cert):
    standin = StandIn(*cert)
    e = Env(tmp_path / "root", standin, cert[0])
    yield e
    standin.close()


def _redirecting(env, seen):
    """The real https upstream, with the connection moved from the default port to the stand-in."""

    def up(**kw):
        seen.append((kw["host"], kw["port"]))
        return https_upstream(**{**kw, "port": env.standin.port})

    return up


@pytest.mark.parametrize("spelling", ["https://127.0.0.1", "https://127.0.0.1/", "HTTPS://127.0.0.1:443"])
def test_default_port_hosted_entry_is_served_for_the_brokers_canonical_form(env, spelling):
    env.write_config(providers={"anth": env.provider(base_url=spelling)})
    seen = []
    h = env.holder(clock=Clock(), upstream=_redirecting(env, seen))
    reply = call(h, endpoint="https://127.0.0.1:443", body=body_for(100))
    assert reply.authenticated and reply.status == 200, reply.refusal_code
    assert seen == [("127.0.0.1", 443)]
    assert len(env.standin.seen) == 1
    # the raw spelling a service would send maps to the same provider
    assert call(h, endpoint=spelling, body=body_for(100)).status == 200
    h.close()


def test_a_different_origin_is_still_unknown(env):
    env.write_config(providers={"anth": env.provider(base_url="https://127.0.0.1")})
    h = env.holder(clock=Clock(), upstream=_redirecting(env, []))
    for endpoint in ("https://127.0.0.1:8443", "http://127.0.0.1:443", "https://127.0.0.2:443", "nonsense", ""):
        assert call(h, endpoint=endpoint).refusal_code == "endpoint_unknown"
    h.close()


def _cfg(env, providers):
    return GatewayConfig(
        protected=True,
        providers=providers,
        holder=env.holder_table(),
        run_bounds=(),
    )


def test_two_providers_with_one_canonical_origin_fail_the_start(env):
    cfg = _cfg(
        env,
        {
            "a": env.provider(base_url="https://api.example.com"),
            "b": env.provider(base_url="HTTPS://API.example.com:443/"),
        },
    )
    with pytest.raises(HolderStartError, match="share the origin"):
        build_policy(cfg)


def test_an_uncanonicalisable_base_url_fails_the_start(env):
    cfg = _cfg(env, {"a": env.provider(base_url="ftp://api.example.com")})
    with pytest.raises(HolderStartError, match="invalid base_url"):
        build_policy(cfg)


# --- D7b: the broker's reply cap and the resolved address in calls.log ---------------------------


def test_the_broker_reply_cap_constant_is_pinned_to_ccg_source():
    from llmgateway.keyholder import MAX_BROKER_RESPONSE_BYTES

    src = (CCG_MANIFEST.parent / "broker" / "framing.py").read_text()
    value = next(
        eval(compile(ast.Expression(n.value), "<ccg>", "eval"), {})  # noqa: S307 - a constant product
        for n in ast.parse(src).body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "MAX_RESPONSE_BYTES"
    )
    assert MAX_BROKER_RESPONSE_BYTES == value == 16 * 1024 * 1024


@pytest.mark.parametrize("size,ok", [(16 * 1024 * 1024, True), (16 * 1024 * 1024 + 1, False)])
def test_max_response_bytes_above_the_broker_cap_fails_the_start(env, size, ok):
    cfg = GatewayConfig(
        protected=True,
        providers={"a": env.provider()},
        holder=env.holder_table(max_response_bytes=size),
        run_bounds=(),
    )
    if ok:
        assert build_policy(cfg).max_response == size
    else:
        with pytest.raises(HolderStartError, match="max_response_bytes"):
            build_policy(cfg)


def test_calls_log_records_the_address_connected_to_or_null(env):
    import json

    h = env.holder(clock=Clock(), upstream=_redirecting(env, []))
    assert call(h, endpoint=env.endpoint).status == 200
    assert call(h, endpoint="https://nowhere.example:1").refusal_code == "endpoint_unknown"
    rows = [json.loads(x) for x in open(f"{env.state_root}/calls.log")]
    assert [r["resolved_addr"] for r in rows] == ["127.0.0.1", None]
    h.close()
