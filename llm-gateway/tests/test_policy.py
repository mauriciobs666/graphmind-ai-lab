"""Scope policy and chain resolution (plan §3.6; FR-3, FR-6, AC-2, AC-4, AC-17, AC-21)."""

import os

import pytest

from llmgateway.config import ConfigSource, GatewayConfig
from llmgateway.policy import PolicyRefused, check_run_bound, resolve
from llmgateway.records import Attempt, RecordStore
from llmgateway.types import FailureClass as F
from llmgateway.types import Request

ME = os.getuid()

TOML = """
[deployment]
local_endpoints = ["http://127.0.0.1:1234"]

[providers.lm]
kind = "lmstudio"
base_url = "http://127.0.0.1:1234/v1"

[providers.claude]
kind = "anthropic"
base_url = "https://api.anthropic.com"

[providers.remote]
kind = "openai_compat"
base_url = "https://203.0.113.5/v1"

[scopes.plain]
[scopes.open]
marker = "unrestricted"
record_text = true
[scopes.secret]
marker = "restricted"
record_text = true
[scopes.texty]
record_text = true
[scopes."run:big"]
[scopes."run:mix"]

[scope_sources.ccg-manifest]
prefix = "ccg:"

[call_sites.local]
component = "answer.local"
chain = ["lm"]

[call_sites.hosted]
component = "answer.hosted"
scopes = ["plain"]
chain = ["claude", "lm"]
local_chain = ["lm"]

[call_sites.nolocal]
component = "answer.nolocal"
scopes = ["plain"]
chain = [{ provider = "claude", model = "claude-x" }]

[call_sites.ccg]
component = "answer.interpret"
scopes = ["ccg:system:main"]
chain = ["claude"]
local_chain = ["lm"]

[call_sites.remote]
component = "answer.remote"
scopes = ["plain"]
chain = ["remote"]

[call_sites.mixed]
component = "answer.mixed"
scopes = ["plain"]
chain = ["claude", "lm"]

[call_sites.badlocal]
component = "answer.badlocal"
scopes = ["plain"]
chain = ["claude"]
local_chain = ["claude"]

[call_sites.noscope]
component = "answer.noscope"
chain = ["claude"]

[[scope_overrides]]
scope = "run:big"
call_site = "hosted"
chain = ["claude"]

[[scope_overrides]]
scope = "run:big"
chain = ["remote"]

[[scope_overrides]]
scope = "run:mix"
chain = ["claude", "lm"]

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


def load(root, text=TOML, trusted=frozenset({ME}), mode=0o644):
    f = root / "etc" / "gateway.toml"
    f.write_text(text)
    f.chmod(mode)
    return ConfigSource(f, trusted, anchor=root).get()


@pytest.fixture
def cfg(root):
    return load(root)


class Markers:
    """A recording ScopeMarkerSource; `answers` maps scope id -> marker or an exception."""

    def __init__(self, **answers):
        self.answers = {k.replace("_", ":"): v for k, v in answers.items()}
        self.reads = []

    def marker(self, scope_id):
        self.reads.append(scope_id)
        value = self.answers.get(scope_id, "unrestricted")
        if isinstance(value, Exception):
            raise value
        return value


def req(site="hosted", *scopes, **kw):
    return Request(
        call_site=site, messages=({"role": "user", "content": "hi"},), scopes=scopes, **kw
    )


def providers(res):
    return [s.provider for s in res.chain]


# --- FR-6 / AC-4: defaults are local ------------------------------------------------------


def test_local_chain_needs_no_scope_and_reads_no_hosted_gate(cfg):
    res = resolve(cfg, req("local"))
    assert providers(res) == ["lm"] and not res.has_hosted and res.source == "default"


def test_local_call_accepts_unregistered_wellformed_scopes(cfg):
    res = resolve(cfg, req("local", "whatever:else"))
    assert res.scopes == ("whatever:else",) and not res.has_hosted


def test_unprotected_config_disables_hosted_and_filters_nothing_silently(root):
    cfg = load(root, mode=0o664)  # group-writable: fails the owner check
    assert not cfg.protected
    assert providers(resolve(cfg, req("hosted"))) == ["lm"]  # loader dropped the hosted step
    for site in ("nolocal", "remote", "noscope"):  # chain emptied: refused, not run elsewhere
        with pytest.raises(PolicyRefused) as e:
            resolve(cfg, req(site))
        assert e.value.reason == "empty_chain" and e.value.failure_class is F.CONFIG


def test_hosted_step_in_an_unprotected_config_is_refused_directly():
    cfg = GatewayConfig(
        protected=False,
        providers={"claude": {"kind": "anthropic", "base_url": "https://api.anthropic.com"}},
        call_sites={"s": {"chain": ["claude"]}},
    )
    with pytest.raises(PolicyRefused) as e:
        resolve(cfg, req("s", "plain"))
    assert e.value.reason == "hosted_disabled"


def test_remote_openai_compat_url_is_hosted_and_gated(cfg):
    res = resolve(cfg, req("remote"))
    assert res.has_hosted  # hosted, so it passed only because scope `plain` is registered
    with pytest.raises(PolicyRefused) as e:
        resolve(cfg, req("remote", "not:registered"))
    assert e.value.reason == "scope_unregistered"


# --- registration ------------------------------------------------------------------------


def test_hosted_needs_a_registered_scope(cfg):
    with pytest.raises(PolicyRefused) as e:
        resolve(cfg, req("noscope"))
    assert e.value.reason == "scope_required" and e.value.failure_class is F.POLICY_REFUSED


def test_unregistered_scope_refuses_hosted_but_not_local(cfg):
    with pytest.raises(PolicyRefused) as e:
        resolve(cfg, req("hosted", "unknown:scope"))
    assert e.value.reason == "scope_unregistered"
    assert providers(resolve(cfg, req("local", "unknown:scope"))) == ["lm"]


@pytest.mark.parametrize(
    "scope",
    [
        "Workspace:acme",
        "plain ",
        "plaіn",
        "plаin",
        "plain\n",
        "a b",
        "",
        "x" * 129,
        "ccg:repo:" + "a" * 120,
        "ccg:repo:сcg",
    ],
)
def test_confusable_or_malformed_scope_ids_are_refused(cfg, scope):
    for site in ("hosted", "local"):
        with pytest.raises(PolicyRefused) as e:
            resolve(cfg, req(site, scope))
        assert e.value.reason == "scope_malformed"


def test_scope_id_of_128_chars_is_wellformed_and_source_backed(cfg):
    sid = "ccg:repo:" + "a" * (128 - 9)
    assert len(sid) == 128
    assert resolve(cfg, req("ccg", sid), Markers()).scopes[-1] == sid


# --- restricted scopes, sticky rule, local_chain ------------------------------------------


def test_restricted_scope_with_hosted_default_goes_to_local_chain(cfg):
    res = resolve(cfg, req("hosted", "secret"))
    assert providers(res) == ["lm"] and res.source == "local_chain" and not res.has_hosted


def test_restricted_scope_without_local_chain_is_refused(cfg):
    with pytest.raises(PolicyRefused) as e:
        resolve(cfg, req("nolocal", "secret"))
    assert e.value.reason == "restricted_hosted"


def test_restricted_scope_with_hosted_override_is_refused_even_with_local_chain(cfg):
    with pytest.raises(PolicyRefused) as e:
        resolve(cfg, req("hosted", "run:big", "secret"))
    assert e.value.reason == "restricted_hosted_override"


def test_sticky_one_restricted_scope_among_many_sends_the_call_local(cfg):
    res = resolve(cfg, req("hosted", "open", "secret", "texty"))
    assert providers(res) == ["lm"] and res.source == "local_chain"


def test_adding_unrestricted_scopes_cannot_loosen_a_restricted_one(cfg):
    base = resolve(cfg, req("ccg"), Markers(ccg_system_main="restricted"))
    more = resolve(cfg, req("ccg", "open", "plain", "texty"), Markers(ccg_system_main="restricted"))
    assert providers(base) == providers(more) == ["lm"]
    assert set(base.scopes) <= set(more.scopes)  # a request only adds


def test_a_request_can_tighten_with_a_restricted_scope(cfg):
    assert resolve(cfg, req("hosted")).has_hosted
    assert not resolve(cfg, req("hosted", "secret")).has_hosted


def test_request_scopes_never_remove_a_bound_scope(cfg):
    res = resolve(cfg, req("hosted", "open"))
    assert res.scopes == ("plain", "open")
    assert resolve(cfg, req("hosted", "plain")).scopes == ("plain",)  # no duplicate either


def test_local_chain_naming_a_hosted_step_is_refused(cfg):
    with pytest.raises(PolicyRefused) as e:
        resolve(cfg, req("badlocal", "secret"))
    assert e.value.reason == "local_chain_hosted"


def test_registered_scope_without_marker_is_unrestricted(cfg):
    res = resolve(cfg, req("hosted"))
    assert res.has_hosted and res.markers == {"plain": "unrestricted"}


# --- markers: unreadable / ill-typed / per call, uncached ----------------------------------


def test_source_marker_restricted_goes_local_and_unrestricted_goes_hosted(cfg):
    assert not resolve(cfg, req("ccg"), Markers(ccg_system_main="restricted")).has_hosted
    assert resolve(cfg, req("ccg"), Markers(ccg_system_main="unrestricted")).has_hosted


def test_unreadable_source_marker_refuses_hosted(cfg):
    for source in (
        Markers(ccg_system_main=OSError("gone")),
        Markers(ccg_system_main=KeyError("x")),
        None,
    ):
        with pytest.raises(PolicyRefused) as e:
            resolve(cfg, req("ccg"), source)
        assert e.value.reason == "marker_unreadable"


@pytest.mark.parametrize(
    "bad",
    [None, True, 1, "Restricted", "restricted ", "local-only", "", b"restricted", ["restricted"]],
)
def test_ill_typed_source_marker_refuses_hosted(cfg, bad):
    with pytest.raises(PolicyRefused) as e:
        resolve(cfg, req("ccg"), Markers(ccg_system_main=bad))
    assert e.value.reason == "marker_unreadable"


def test_unreadable_marker_is_not_treated_as_restricted_fallback(cfg):
    # it refuses outright (never silently reroutes to local_chain, which is configured here)
    with pytest.raises(PolicyRefused):
        resolve(cfg, req("ccg"), Markers(ccg_system_main=ValueError("bad")))


def test_unreadable_marker_does_not_block_an_all_local_call(cfg):
    res = resolve(cfg, req("local", "ccg:repo:x"), Markers(ccg_repo_x=OSError("gone")))
    assert providers(res) == ["lm"]


def test_marker_is_read_on_every_call_and_tightening_takes_effect_next_call(cfg):
    src = Markers(ccg_system_main="unrestricted")
    assert resolve(cfg, req("ccg"), src).has_hosted
    src.answers["ccg:system:main"] = "restricted"
    assert not resolve(cfg, req("ccg"), src).has_hosted
    src.answers["ccg:system:main"] = "unrestricted"
    assert resolve(cfg, req("ccg"), src).has_hosted
    assert src.reads == ["ccg:system:main"] * 3


def test_static_ill_typed_marker_refuses_hosted():
    cfg = GatewayConfig(
        protected=True,
        providers={
            "claude": {"kind": "anthropic", "base_url": "https://api.anthropic.com"},
            "lm": {"kind": "lmstudio", "base_url": "http://127.0.0.1:1"},
        },
        scopes={"s": {"marker": "Restricted"}},
        call_sites={"c": {"scopes": ["s"], "chain": ["claude"], "local_chain": ["lm"]}},
        local_origins=frozenset({("http", "127.0.0.1", 1)}),
    )
    with pytest.raises(PolicyRefused) as e:
        resolve(cfg, req("c"))
    assert e.value.reason == "marker_unreadable"


def test_static_marker_under_a_source_prefix_is_not_believed():
    cfg = GatewayConfig(
        protected=True,
        providers={"claude": {"kind": "anthropic", "base_url": "https://api.anthropic.com"}},
        scopes={"ccg:x": {"marker": "unrestricted"}},
        scope_sources={"m": {"prefix": "ccg:"}},
        call_sites={"c": {"scopes": ["ccg:x"], "chain": ["claude"]}},
    )
    with pytest.raises(PolicyRefused) as e:
        resolve(cfg, req("c"), Markers())
    assert e.value.reason == "marker_unreadable"


def test_request_cannot_carry_a_marker_or_a_model():
    for extra in (
        {"marker": "unrestricted"},
        {"model": "x"},
        {"provider": "claude"},
        {"chain": ["claude"]},
    ):
        with pytest.raises(ValueError, match="unknown request fields"):
            Request.from_dict({"call_site": "hosted", "messages": [], **extra})


# --- chains: overrides, precedence, step shapes -------------------------------------------


def test_override_selects_the_chain_in_file_order_first_match_wins(cfg):
    res = resolve(cfg, req("hosted", "run:big"))
    assert res.source == "override" and providers(res) == ["claude"]
    other = resolve(cfg, req("remote", "run:big"))  # first entry is for call site `hosted` only
    assert other.source == "override" and providers(other) == ["remote"]


def test_no_override_without_a_matching_scope(cfg):
    assert resolve(cfg, req("hosted")).source == "default"


def test_override_hosted_chain_is_allowed_for_unrestricted_registered_scopes(cfg):
    assert resolve(cfg, req("local", "run:big")).has_hosted


def test_unprotected_config_ignores_overrides(root):
    cfg = load(root, mode=0o666)
    assert resolve(cfg, req("hosted", "run:big")).source == "default"


def test_step_shapes_string_and_mapping(cfg):
    res = resolve(cfg, req("nolocal"))
    assert res.chain[0].provider == "claude" and res.chain[0].model == "claude-x"
    assert res.chain[0].hosted
    res = resolve(cfg, req("hosted"))
    assert [(s.provider, s.hosted) for s in res.chain] == [("claude", True), ("lm", False)]


def test_unknown_call_site_and_unknown_provider_are_config_errors(cfg):
    with pytest.raises(PolicyRefused) as e:
        resolve(cfg, req("nope"))
    assert e.value.reason == "unknown_call_site" and e.value.failure_class is F.CONFIG
    bad = GatewayConfig(protected=True, call_sites={"c": {"chain": ["ghost"]}})
    with pytest.raises(PolicyRefused) as e:
        resolve(bad, req("c"))
    assert e.value.reason == "unknown_provider"
    bad = GatewayConfig(protected=True, call_sites={"c": {"chain": [{"model": "m"}]}})
    with pytest.raises(PolicyRefused):
        resolve(bad, req("c"))


# --- text recording (AC-17) --------------------------------------------------------------


def test_text_recording_needs_every_scope_opted_in_and_none_restricted(cfg):
    assert resolve(cfg, req("local", "open")).record_text
    assert resolve(cfg, req("local", "open", "texty")).record_text
    assert not resolve(cfg, req("local", "open", "plain")).record_text  # plain has no opt-in
    assert not resolve(cfg, req("local", "secret")).record_text  # opted in but restricted
    assert not resolve(cfg, req("local", "open", "secret")).record_text
    assert not resolve(cfg, req("local")).record_text  # no scope at all
    assert not resolve(cfg, req("local", "unreg:scope")).record_text


def test_text_recording_off_when_a_source_marker_is_unreadable_or_restricted(root):
    cfg = load(
        root,
        TOML.replace(
            '[scopes."run:big"]',
            '[scopes."ccg:system:main"]\nrecord_text = true\n[scopes."run:big"]',
        ),
    )
    assert resolve(cfg, req("ccg"), Markers(ccg_system_main="unrestricted")).record_text
    assert not resolve(cfg, req("ccg"), Markers(ccg_system_main="restricted")).record_text
    assert not resolve(
        cfg, req("local", "ccg:system:main"), Markers(ccg_system_main=OSError())
    ).record_text


def test_restricted_resolution_never_allows_text_into_the_store(cfg, tmp_path):
    store = RecordStore(tmp_path / "s")
    res = resolve(cfg, req("local", "secret"))
    store.write(
        Attempt(
            call_id="c",
            attempt=1,
            component=res.component,
            scopes=res.scopes,
            call_site=res.call_site,
            provider="lm",
            model="m",
            hosted=False,
            outcome="ok",
            request_text="SECRET PROMPT",
            response_text="SECRET ANSWER",
        ),
        text_allowed=res.record_text,
    )
    row = store.attempts("c")[0]
    assert row["request_text"] is None and row["response_text"] is None
    res = resolve(cfg, req("local", "open"))
    store.write(
        Attempt(
            call_id="d",
            attempt=1,
            component=res.component,
            scopes=res.scopes,
            call_site=res.call_site,
            provider="lm",
            model="m",
            hosted=False,
            outcome="ok",
            request_text="p",
            response_text="a",
        ),
        text_allowed=res.record_text,
    )
    assert store.attempts("d")[0]["request_text"] == "p"


# --- library-side run bound (§3.9) -------------------------------------------------------


def _spend(store, run_id, n, cid):
    store.write(
        Attempt(
            call_id=cid,
            attempt=1,
            component="c",
            scopes=(),
            call_site="s",
            provider="lm",
            model="m",
            hosted=False,
            outcome="ok",
            run_id=run_id,
            input_tokens=n,
            output_tokens=0,
        )
    )


def test_run_bound_refuses_at_the_bound_only_for_matching_runs(cfg, tmp_path):
    store = RecordStore(tmp_path / "s")
    check_run_bound(cfg, "eval-1", store)  # nothing used
    _spend(store, "eval-1", 99, "a")
    check_run_bound(cfg, "eval-1", store)  # under
    _spend(store, "eval-1", 1, "b")
    with pytest.raises(PolicyRefused) as e:
        check_run_bound(cfg, "eval-1", store)
    assert e.value.reason == "run_bound_reached" and e.value.failure_class is F.RUN_BOUND
    _spend(store, "interactive", 10**6, "c")
    check_run_bound(cfg, "interactive", store)  # matches no entry: unbounded
    check_run_bound(cfg, None, store)
    check_run_bound(cfg, "eval-2", store)  # a different run's counter


def test_run_bound_with_invalid_limit_is_a_config_error(tmp_path):
    cfg = GatewayConfig(protected=True, run_bounds=({"match": "e*", "max_tokens": "10"},))
    with pytest.raises(PolicyRefused) as e:
        check_run_bound(cfg, "e1", RecordStore(tmp_path / "s"))
    assert e.value.failure_class is F.CONFIG


def test_unprotected_config_has_no_run_bounds(root, tmp_path):
    cfg = load(root, mode=0o666)
    store = RecordStore(tmp_path / "s")
    _spend(store, "eval-1", 10**6, "a")
    check_run_bound(cfg, "eval-1", store)


def test_hosted_steps_are_never_silently_filtered_from_a_mixed_chain(cfg):
    # the argued-down alternative: drop the hosted step and run the local remainder
    with pytest.raises(PolicyRefused) as e:
        resolve(cfg, req("mixed", "secret"))
    assert e.value.reason == "restricted_hosted"
    with pytest.raises(PolicyRefused) as e:
        resolve(cfg, req("local", "run:mix", "secret"))
    assert e.value.reason == "restricted_hosted_override"
    # unrestricted: the whole chain, hosted step included, is returned untouched
    assert [s.provider for s in resolve(cfg, req("mixed")).chain] == ["claude", "lm"]


def test_override_entry_without_a_chain_is_a_config_error_not_the_default_chain(root):
    text = TOML + '\n[[scope_overrides]]\nscope = "texty"\n'
    cfg = load(root, text)
    with pytest.raises(PolicyRefused) as e:
        resolve(cfg, req("hosted", "texty"))
    assert e.value.failure_class is F.CONFIG and e.value.reason == "empty_chain"
    assert resolve(cfg, req("hosted")).source == "default"  # other scopes are untouched


@pytest.mark.parametrize("chain", ['"lm"', "[]", "7"])
def test_override_entry_with_an_unusable_chain_is_a_config_error(root, chain):
    cfg = load(root, TOML + f'\n[[scope_overrides]]\nscope = "texty"\nchain = {chain}\n')
    with pytest.raises(PolicyRefused) as e:
        resolve(cfg, req("local", "texty"))
    assert e.value.failure_class is F.CONFIG
