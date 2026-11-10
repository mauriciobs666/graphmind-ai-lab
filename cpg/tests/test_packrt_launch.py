"""P1-16 `ccg.packrt.launch`: the `fastmcp` pack's launch declarations over the paper records of rule-pack spec §4.3.

Plan: `docs/plans/connected-code-graph-p1-linking.md` § P1-16 (the `fastmcp` row of rule-pack spec §12.1: the `(8200, unit)`
port and the `/mcp` mount path of §4.4). The records are written from the spec's table (L1-L5, C1) with the argv gate's
output as the spec states it, so the cases are the spec's own; extra cases plant the edges the table does not reach.
"""
import copy

import pytest

from packrt_helpers import call, cond, ctrl, data, file_, launch, pack, sym, vs
from ccg.bundle import REDACTED
from ccg.packrt import run_packs

FASTMCP = pack("fastmcp")
APP = "falkor-chat/server/falkorchat/app.py"
START = "falkor-chat/scripts/start_agent_team.sh"


def paper(**over):
    """§4.3's L1-L5: the Dockerfile `uvicorn` (L1), two redacted script launches (L2, L4), two env-default facts (L3, L5)."""
    recs = [
        file_("fa", APP, unit="U-server"), file_("fb", "falkor-chat/server/falkorchat/other.py", unit="U-server"),
        launch("L1", "falkor-chat/Dockerfile", "command", command="uvicorn",
               args=["falkorchat.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "/app/config/logging.json"]),
        launch("L2", "falkor-chat/scripts/start_server.sh", "command", command="<redacted>",
               args=["falkorchat.app:app", "<redacted>", "<redacted>"]),
        launch("L3", START, "env-default", envVar="AGENT_TEAM_PORT", default="8200", origin="env-default"),
        launch("L4", START, "command", command="<redacted>", args=["falkorchat.app:app", "<redacted>", "<redacted>"]),
        launch("L5", START, "env-default", envVar="FALKORDB_PORT", default="6379", origin="env-default"),
    ]
    return recs + list(over.get("extra", ()))


def ports(out):
    return sorted((f["unitId"], f["port"], f["origin"]) for f in out.launch_facts if f["fact"] == "port")


def run(recs, pack=FASTMCP):
    return run_packs([recs], [pack])


def test_spec_4_4_gives_the_explicit_8000_and_the_env_default_8200_and_not_the_excluded_6379():
    assert ports(run(paper())) == [("U-server", 8000, "explicit"), ("U-server", 8200, "env-default")]


def test_the_launched_unit_is_the_one_in_repo_file_that_the_module_attr_argument_names():
    out = run(paper())
    got = {f["launchId"]: (f["unitId"], f["target"]) for f in out.launch_facts if f["fact"] == "launchUnit"}
    assert got == {i: ("U-server", "falkorchat.app:app") for i in ("L1", "L2", "L4")}   # the redacted L2/L4 qualify: no command constraint
    ambiguous = paper(extra=[file_("fx", "other/falkorchat/app.py", unit="U-other")])
    assert [f for f in run(ambiguous).launch_facts if f["fact"] in ("launchUnit", "port")] == []   # several matches resolve nothing
    none = [r for r in paper() if r["type"] != "file"]
    assert run(none).launch_facts == []                                                          # zero matches resolve nothing


def test_the_env_default_port_is_adopted_only_by_a_command_launch_in_the_same_file():
    recs = [r for r in paper() if r["id"] != "L4"]                          # no same-file command launch: nothing to adopt
    assert ports(run(recs)) == [("U-server", 8000, "explicit")]
    moved = [{**r, "path": "elsewhere.sh"} if r["id"] == "L3" else r for r in paper()]
    assert ports(run(moved)) == [("U-server", 8000, "explicit")]


def test_a_command_launch_without_a_port_flag_takes_the_framework_default_as_pack_default():
    recs = [r if r["id"] != "L1" else launch("L1", "Dockerfile", "command", command="/venv/bin/uvicorn", args=["falkorchat.app:app"]) for r in paper()]
    assert ports(run(recs)) == [("U-server", 8000, "pack-default"), ("U-server", 8200, "env-default")]


def test_a_redacted_command_gets_no_default_because_the_basename_cannot_match():
    recs = [r for r in paper() if r["id"] not in ("L1", "L3", "L5")]
    assert ports(run(recs)) == []


@pytest.mark.parametrize("hidden", [REDACTED, "--" + REDACTED, "-" + REDACTED])
def test_a_visible_command_with_a_redacted_argument_gets_no_pack_default_because_the_hidden_word_may_be_a_port_flag(hidden):
    """Spec 1.6/4.4: the default reads emitted text, sound only where command and args are visible; `$EXTRA_ARGS` may be `--port 9000`."""
    recs = [r if r["id"] != "L1" else launch("L1", "Dockerfile", "command", command="uvicorn", args=["falkorchat.app:app", hidden])
            for r in paper() if r["id"] not in ("L3", "L5")]
    assert ports(run(recs)) == []
    assert any(f["fact"] == "launchUnit" and f["launchId"] == "L1" for f in run(recs).launch_facts)    # the unit is still known


def test_the_same_launch_with_every_argument_visible_and_no_port_flag_still_gets_the_pack_default():
    recs = [r if r["id"] != "L1" else launch("L1", "Dockerfile", "command", command="uvicorn", args=["falkorchat.app:app", "--reload"])
            for r in paper() if r["id"] not in ("L3", "L5")]
    assert ports(run(recs)) == [("U-server", 8000, "pack-default")]


def test_an_explicit_port_flag_is_still_read_beside_a_redacted_argument():
    recs = [r if r["id"] != "L1" else launch("L1", "Dockerfile", "command", command="uvicorn",
                                             args=["falkorchat.app:app", "--port", "9001", REDACTED])
            for r in paper() if r["id"] not in ("L3", "L5")]
    assert ports(run(recs)) == [("U-server", 9001, "explicit")]


def test_a_port_derived_from_a_framework_command_carries_the_frameworks_protocol():
    recs = [r if r["id"] != "L1" else launch("L1", "Dockerfile", "command", command="uvicorn", args=["falkorchat.app:app"]) for r in paper()]
    by_origin = {f["origin"]: f.get("protocol") for f in run(recs).launch_facts if f["fact"] == "port"}
    assert by_origin == {"pack-default": "http", "env-default": None}       # L4's command is redacted: no framework known
    explicit = [f for f in run(paper()).launch_facts if f["fact"] == "port" and f["origin"] == "explicit"]
    assert [f["protocol"] for f in explicit] == ["http"]


@pytest.mark.parametrize("args,want", [
    (["falkorchat.app:app", "--port=9001"], [("U-server", 9001, "explicit")]),
    (["falkorchat.app:app", "--port", "9002"], [("U-server", 9002, "explicit")]),
    (["falkorchat.app:app", "--port", "abc"], []),                          # a flag is present: no default either
    (["falkorchat.app:app", "--port", "0"], []),
    (["falkorchat.app:app", "--port", "70000"], []),
    (["falkorchat.app:app", "--port"], []),
])
def test_port_flags_read_both_spellings_and_reject_non_ports(args, want):
    recs = [r if r["id"] != "L1" else launch("L1", "Dockerfile", "command", command="uvicorn", args=args)
            for r in paper() if r["id"] not in ("L3", "L5")]
    assert ports(run(recs)) == want


def test_a_dependency_port_is_excluded_by_name_and_the_start_anchored_limit_is_pinned():
    extra = [launch("L6", START, "env-default", envVar="AGENT_TEAM_FALKORDB_PORT", default="6380", origin="env-default"),
             launch("L7", START, "env-default", envVar="REDIS_PORT", default="6381", origin="env-default"),
             launch("L8", START, "env-default", envVar="HOME", default="1", origin="env-default")]
    got = ports(run(paper(extra=extra)))
    assert ("U-server", 6380, "env-default") in got                          # documented soundness limit (spec 4.4)
    assert not any(p in (6379, 6381, 1) for _, p, _ in got)


def test_a_pack_without_exclude_vars_adopts_a_dependency_port():
    pack = copy.deepcopy(FASTMCP)
    del pack["launch"]["envPorts"][0]["excludeVars"]
    assert ("U-server", 6379, "env-default") in ports(run(paper(), pack))


def test_a_non_numeric_env_default_is_not_a_port():
    recs = [r if r["id"] != "L3" else launch("L3", START, "env-default", envVar="AGENT_TEAM_PORT", default="${X}", origin="env-default") for r in paper()]
    assert ("U-server", 8200, "env-default") not in ports(run(recs))


# ---- mount paths

def mount_records(*, path="/mcp", app_target="mcp/server/fastmcp.py:<module>.FastMCP.streamable_http_app"):
    return [
        file_("fa", APP, unit="U-server"), sym("mk", "create_app", fid="fa", path=APP),
        call("sh", "mk", "streamable_http_app", resolution="external", external=app_target, line=413),
        call("mt", "mk", "mount", line=543), vs("mt", "a0", path), data("mt", "sh", "a1", "r"),
        call("st", "mk", "StaticFiles", resolution="external", external="starlette/staticfiles.py:<module>.StaticFiles"),
        call("mt2", "mk", "mount", line=551), vs("mt2", "a0", "/shop"), data("mt2", "st", "a1", "r"),
        cond("k", "mk", "mount_mcp", line=542), ctrl("mt", "k"),
    ]


def test_a_mount_whose_app_comes_from_a_fastmcp_streamable_app_declares_the_units_mcp_path_under_its_condition():
    (m,) = [f for f in run(mount_records()).launch_facts if f["fact"] == "mountPath"]
    assert (m["unitId"], m["path"], m["pathMatch"], m["siteId"], m["line"]) == ("U-server", "/mcp", "prefix", "mt", 543)
    assert m["conditions"] == [{"siteId": "mt", "line": 542, "code": "mount_mcp"}]


def test_the_inner_endpoint_override_is_never_read_so_matching_stays_prefix_only():
    """O-5: `streamable_http_path = "/"` is an attribute write no record carries; the fact says `prefix`, nothing more."""
    (m,) = [f for f in run(mount_records()).launch_facts if f["fact"] == "mountPath"]
    assert m["pathMatch"] == "prefix" and "innerPath" not in m


def test_a_mount_of_some_other_app_and_an_unresolved_path_declare_nothing():
    assert [f for f in run(mount_records(app_target="x/y.py:<module>.Other.app")).launch_facts if f["fact"] == "mountPath"] == []
    assert [f for f in run(mount_records(path={"unknown": "point-derived"})).launch_facts if f["fact"] == "mountPath"] == []


def test_a_pack_with_no_launch_section_yields_no_launch_facts():
    pack = {k: v for k, v in FASTMCP.items() if k != "launch"}
    assert run(paper() + mount_records(), pack).launch_facts == []


def test_a_target_arg_entry_may_constrain_its_command():
    pack = copy.deepcopy(FASTMCP)
    pack["launch"]["targetArg"][0]["command"] = {"basename": "uvicorn"}
    got = {f["launchId"] for f in run(paper(), pack).launch_facts if f["fact"] == "launchUnit"}
    assert got == {"L1"}                                                    # the redacted L2/L4 no longer qualify (spec 4.4 step 1)


def test_a_mount_call_in_a_file_with_no_unit_declares_nothing():
    recs = [r for r in mount_records() if r["type"] != "file"]
    assert [f for f in run(recs).launch_facts if f["fact"] == "mountPath"] == []
