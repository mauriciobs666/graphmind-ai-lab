"""P1-54: the Binder's environment-read recognition on **real** records (``docs/plans/connected-code-graph-p1-linking.md`` P1-54; P1-17 behaviour (5)).

The inputs are callsite, ``valueSet`` and ``data`` records extracted from a real Joern build (``tests/data/p1_54_environ_reads.json``: the fixture
system's ``fx-svc`` with ``clients/peer.py`` rewritten to ten shapes, plus ``clients/jobs.py`` as committed), never hand-built: the hand-built
records of ``test_linker_bind.py`` once passed a predicate that no real record satisfied. Only the owning symbol id is rewritten (to the
readable ``C``) so the records sit in the ``compose_world`` units.
"""
import copy
import json
from pathlib import Path

import pytest

from ccg.linker import bind as bind_mod
from linker_bind_helpers import config_entry, make_binder, run_link, url_lifted
from linker_helpers import sym_in
from test_linker_bind import bound_by, compose_world, consumer, one

GOLDEN = json.loads((Path(__file__).parent / "data" / "p1_54_environ_reads.json").read_text())
SHAPES = GOLDEN["shapes"]
HTTP_TARGET = "requests.py:<module>.get"
ENV_GET = "os.py:<module>.<member>(environ).get"
FX_PEER_BASE = config_entry("E1", "deploy/compose.yaml", "services.web.environment.FX_PEER_BASE", bind="http://api:8000")

#: shape -> what the real extractor wrote for the environment read that flows into the URL slot
READ_TARGET = {"shape_a": ENV_GET, "shape_b": ENV_GET, "shape_c": ENV_GET, "shape_d": "os.py:<module>.getenv", "shape_e": "__builtin.dict.get",
               "shape_f": ENV_GET, "shape_g": ENV_GET + ".rstrip", "shape_h": "os.py:<module>.<member>(environ).setdefault",
               "shape_i": "os.py:<module>.environ.get", "shape_j": "os.py:<module>.getenv"}


def world(shape, *, compose=(), path="/api/health"):
    """A consumer whose URL slot is fed by the shape's real environment-read record: its binder and the site id."""
    facts, bundles = compose_world(url="", consumer_key=f"GET {path}")
    recs = copy.deepcopy(SHAPES[shape])
    http = next(c for c in recs["callsites"] if c["externalTarget"] == HTTP_TARGET)
    for c in recs["callsites"]:
        c["symbolId"] = "C"
    facts.lifted[:] = [url_lifted(http["id"], "C", "\x00" + path)]
    bundles[0] += list(compose)
    bundles[2] = bundles[2][:1] + [*sym_in("C", "web", path="clients/peer.py"), *recs["callsites"], *recs["valueSets"], *recs["data"]]
    return make_binder(facts, bundles), http["id"], (facts, bundles)


def bound_units(shape, **kw):
    b, site, _ = world(shape, **kw)
    got = b.bind(consumer(site=site, key="GET /api/health"))
    return None if got is None else got.units


def test_the_golden_records_carry_the_shapes_the_plan_names():
    assert {s: next(c["externalTarget"] for c in v["callsites"] if c["externalTarget"] != HTTP_TARGET) for s, v in SHAPES.items()} == READ_TARGET


@pytest.mark.parametrize("shape", ["shape_a", "shape_b", "shape_c", "shape_d", "shape_i", "shape_j"])
def test_a_url_slot_fed_by_a_real_environment_read_binds_the_compose_set_value(shape):
    """`os.environ.get(..) + path`, the f-string, the local, `os.getenv`, and the `from os import` forms all bind `FX_PEER_BASE` to `api`'s unit."""
    assert bound_units(shape, compose=[FX_PEER_BASE]) == ("svc",)


def test_the_environment_read_chain_is_the_compose_entry_then_the_services_mount():
    b, site, (facts, bundles) = world("shape_a", compose=[FX_PEER_BASE])
    facts.providers[1]["key"] = "GET /api/health"
    facts.consumers[0]["siteId"] = site
    link = one(run_link(facts, bundles)[0])
    assert (link.provider, link.reason) == ("P", "binding-confirmed") and bound_by(link) == [(site, 0, "E1"), (site, 1, "M-api")]


def test_the_default_text_of_a_real_environment_read_binds_through_its_compose_port_fact():
    """shape_f: `os.environ.get("FX_PEER_BASE", "http://localhost:8300")`, no Compose value: the default's port is `job`'s published one."""
    assert bound_units("shape_f", path="/job/status") == ("job",)


@pytest.mark.parametrize("shape", ["shape_e", "shape_g", "shape_h"])
def test_a_real_record_that_is_not_an_environment_read_binds_nothing(shape):
    """`dict.get` (same bare `get`), a chained `.get.rstrip` and `.setdefault` are other `externalTarget`s: no Compose value and no default is read."""
    assert bound_units(shape, compose=[FX_PEER_BASE], path="/job/status") is None


def test_a_dict_get_reading_the_same_variable_name_picks_up_no_compose_value():
    """shape_e's `d.get("FX_PEER_BASE", "")` carries the variable name and a Compose entry sets it: only the target tells it from `os.environ.get`."""
    assert bound_units("shape_e", compose=[FX_PEER_BASE]) is None


def test_a_module_constant_read_leaves_no_data_edge_and_stays_unbound_lnk_15():
    """`JOB_URL = os.environ.get(..)` (jobs.py:6): the real environ.get callsite exists but no `data` edge reaches a use, so nothing flows to a URL slot."""
    assert GOLDEN["jobs"]["data"] == []
    assert [c["externalTarget"] for c in GOLDEN["jobs"]["callsites"]] == [ENV_GET]


# ---- the census: the predicate must match real records -----------------------------------------------------------------------
def _real_targets():
    return {c["externalTarget"] for v in (*SHAPES.values(), GOLDEN["jobs"]) for c in v["callsites"]}


def test_census_the_recognised_targets_each_match_a_real_record():
    """A predicate that matches no real record is dead (the P1-17 defect): every table entry is observed in the golden data, none assumed."""
    assert bind_mod._ENV_READ_TARGETS <= _real_targets()
    assert {ENV_GET, "os.py:<module>.getenv", "os.py:<module>.environ.get"} == set(bind_mod._ENV_READ_TARGETS)


def test_census_every_real_environment_read_that_feeds_a_url_slot_is_recognised_exactly_when_it_is_one_of_the_three_reads():
    flows = {d["def"]: next(c for c in v["callsites"] if c["id"] == d["def"])["externalTarget"]
             for v in SHAPES.values() for d in v["data"] if d["toPath"] == "a0"}
    recognised = {t for t in flows.values() if t in bind_mod._ENV_READ_TARGETS}
    assert recognised == set(bind_mod._ENV_READ_TARGETS)
    assert not any(t in bind_mod._ENV_READ_TARGETS for t in ("__builtin.dict.get", ENV_GET + ".rstrip", "os.py:<module>.<member>(environ).setdefault"))


def test_the_binders_predicate_recognises_exactly_the_three_reads_among_every_real_callsite():
    """Over every real callsite of the golden data (chained `.get.rstrip`, `.setdefault`, `dict.get`, `requests.get` among them) and no other."""
    every = [c for v in (*SHAPES.values(), GOLDEN["jobs"]) for c in v["callsites"]]
    assert {c["externalTarget"] for c in every if bind_mod._is_env_read(c)} == {ENV_GET, "os.py:<module>.environ.get", "os.py:<module>.getenv"}
    assert not bind_mod._is_env_read({"calleeName": "environ.get"}) and not bind_mod._is_env_read({"calleeName": "getenv", "resolution": "unresolved"})


def test_the_table_is_exact_strings_not_a_suffix_or_glob():
    assert all(isinstance(t, str) and "*" not in t for t in bind_mod._ENV_READ_TARGETS) and not hasattr(bind_mod, "_ENV_READS")
