"""P1-17 dispatch 2: launch binding (behaviour (6)), the AC-4 ``cypher`` chain: ``.mcp.json`` server -> command -> script -> container -> unit.

Plan: ``docs/plans/connected-code-graph-p1-linking.md`` P1-17 (6); chain as ``BOUND_BY`` edges in chain order (``-p1-graph.md`` §2:
"the server's ``command`` launch entry; the script's ``exec``/``docker run`` launch entry; the container's ``CMD``/``ENTRYPOINT``
launch entry"), ``ord`` 0 nearest the consumer, every edge carrying its ``siteId``. The records are **paper**: no extractor emits a
script's ``exec``/``docker run`` launch fact, and the real ``.mcp.json`` ``command`` carries ``bash -c <redacted>`` (the argv gate
redacts a ``$``-bearing word), which ``test_the_real_extractors_redacted_mcp_command_names_no_script`` pins as a known limit.
"""
import pytest

from linker_bind_helpers import KEY, config_entry, launch_fact, launch_unit, make_binder, pack_facts, run_link, unit_bundle
from linker_helpers import mention, prov, sym_in
from test_linker_bind import bound_by, consumer, one

REPOS = {"r": "."}
CMD_ARGS = ["-c", 'exec "$CLAUDE_PROJECT_DIR/cypher-mcp/docker-run.sh"']


def cypher_world(*, args=CMD_ARGS, command="bash", script="cypher-mcp/docker-run.sh", dockerfile="cypher-mcp/Dockerfile", mcp_path=".mcp.json",
                 mcp_role="primary", extra=(), sites=("m1",)):
    mcp = unit_bundle("mcpcfg", repo="r", root=mcp_path, lang="config", files=[(mcp_path, mcp_role)],
                      records=[launch_fact("L0", mcp_path, "command", service="cypher", command=command, args=args)])
    script_b = unit_bundle("script", repo="r", root=script, lang="shell", files=[(script, "primary")],
                           records=[launch_fact("LS", script, "command", command="docker", args=["run", "--rm", "-i", "cypher-mcp"])])
    docker_b = unit_bundle("dock", repo="r", root=dockerfile, lang="config", files=[(dockerfile, "primary")],
                           records=[launch_fact("LD", dockerfile, "command", command="python", args=["server.py"])])
    cypher = unit_bundle("cypher", repo="r", root="cypher-mcp", records=sym_in("P", "cypher"))
    prompts = unit_bundle("prompts", repo="r", root="claude", lang="markdown",
                          records=[*sym_in("D", "prompts", path="claude/a.md"), *[mention(s, "D", "mcp__cypher__query") for s in sites]])
    facts = pack_facts(providers=[prov("P", "mcp_tool", "query")])
    return facts, [mcp, script_b, docker_b, cypher, prompts, *extra]


def run(facts, bundles):
    return run_link(facts, bundles, repos=REPOS)


def test_the_ac_4_chain_server_command_script_container_unit_is_written_in_chain_order_with_the_site_id_on_every_edge_on_paper_launch_facts():
    got, _ = run(*cypher_world())
    link = one(got)
    assert (link.provider, link.status, link.reason) == ("P", "possible", "prompt-mention")            # a mention is never confirmed
    assert bound_by(link) == [("m1", 0, "L0"), ("m1", 1, "LS"), ("m1", 2, "LD")]


def test_ord_counts_per_link_and_site_not_across_sites_on_paper_launch_facts():
    got, _ = run(*cypher_world(sites=("m1", "m2")))
    assert bound_by(one(got)) == [("m1", 0, "L0"), ("m1", 1, "LS"), ("m1", 2, "LD"), ("m2", 0, "L0"), ("m2", 1, "LS"), ("m2", 2, "LD")]


def test_the_chain_selects_the_unit_of_the_container_and_a_provider_elsewhere_makes_no_link():
    facts, bundles = cypher_world()
    bundles.append(unit_bundle("other", repo="r", root="other", records=sym_in("PO", "other")))
    facts.providers.append(prov("PO", "mcp_tool", "query"))
    got, b = run(facts, bundles)
    assert b.bind(consumer(anchor="D", site="m1", kind="mcp_tool", key="query", server="cypher")).units == ("cypher",)
    assert [l.provider for l in got.links] == ["P"]


def test_a_launch_unit_the_packs_resolved_for_the_script_launch_ends_the_chain_there_on_paper_launch_facts():
    facts, bundles = cypher_world()
    facts.launch_facts.append(launch_unit("LS", "cypher"))
    assert bound_by(one(run(facts, bundles)[0])) == [("m1", 0, "L0"), ("m1", 1, "LS")]


def test_a_launch_unit_the_packs_resolved_for_the_server_command_itself_is_a_one_entry_chain():
    facts, bundles = cypher_world(args=["-c", "nothing"])
    facts.launch_facts.append(launch_unit("L0", "cypher"))
    assert bound_by(one(run(facts, bundles)[0])) == [("m1", 0, "L0")]


def test_a_target_arg_unit_of_the_dockerfile_launch_wins_over_the_directory_rule():
    facts, bundles = cypher_world()
    bundles.append(unit_bundle("elsewhere", repo="r", root="elsewhere", records=sym_in("PE", "elsewhere")))
    facts.providers.append(prov("PE", "mcp_tool", "query"))
    facts.launch_facts.append(launch_unit("LD", "elsewhere"))
    got, _ = run(facts, bundles)
    assert [l.provider for l in got.links] == ["PE"]


def test_the_real_extractors_redacted_mcp_command_names_no_script_so_the_chain_breaks_at_its_first_hop():
    """Known limit, pinned: a ``bash -c <redacted>`` command (any non-allow-listed ``bash -c`` string; P1-11b emits the script path for the allow-listed ``exec "$CLAUDE_PROJECT_DIR/<path>"``) names no script, so it is unbound."""
    facts, bundles = cypher_world(args=["-c", "<redacted>"])
    got, b = run(facts, bundles)
    assert b.bind(consumer(anchor="D", site="m1", kind="mcp_tool", key="query", server="cypher")) is None
    assert bound_by(one(got)) == []


@pytest.mark.parametrize("token", ["/etc/cron.daily/x.sh", "../../outside.sh", "$HOME/cypher-mcp/docker-run.sh", "cypher-mcp/missing.sh",
                                   "`cypher-mcp/docker-run.sh`", "cypher-mcp/docker-run.sh;"])
def test_a_script_path_that_is_absolute_leaves_the_repo_is_unknown_or_is_not_a_plain_path_is_never_followed(token):
    facts, bundles = cypher_world(args=["-c", f"exec {token}"])
    assert make_binder(facts, bundles, repos=REPOS).bind(consumer(anchor="D", site="m1", kind="mcp_tool", key="query", server="cypher")) is None


@pytest.mark.parametrize("arg", ["exec ${CLAUDE_PROJECT_DIR}/cypher-mcp/docker-run.sh", "exec cypher-mcp/docker-run.sh",
                                 "'$CLAUDE_PROJECT_DIR/cypher-mcp/docker-run.sh'"])
def test_the_script_may_be_named_relative_to_the_project_dir_in_any_of_its_lexical_forms(arg):
    facts, bundles = cypher_world(args=["-c", arg])
    got = make_binder(facts, bundles, repos=REPOS).bind(consumer(anchor="D", site="m1", kind="mcp_tool", key="query", server="cypher"))
    assert got is not None and got.units == ("cypher",)


def test_a_script_with_no_dockerfile_beside_it_and_no_launch_unit_ends_the_chain_unbound():
    facts, bundles = cypher_world(dockerfile="Dockerfile")                       # the Dockerfile is at the repo root, not beside the script
    assert make_binder(facts, bundles, repos=REPOS).bind(consumer(anchor="D", site="m1", kind="mcp_tool", key="query", server="cypher")) is None


def test_a_compose_service_of_the_same_name_is_not_a_launch_chain_and_does_not_leak_into_it():
    facts, bundles = cypher_world()
    bundles.append(unit_bundle("comp", repo="r", root="deploy/compose.yaml", lang="config", files=[("deploy/compose.yaml", "primary")], records=[
        launch_fact("MC", "deploy/compose.yaml", "mount", service="cypher", origin="compose", mountPath="../other"),
        launch_fact("PC", "deploy/compose.yaml", "port", service="cypher", origin="compose", port=7000, protocol="tcp")]))
    bundles.append(unit_bundle("other", repo="r", root="other", records=sym_in("PO", "other")))
    facts.providers.append(prov("PO", "mcp_tool", "query"))
    got, _ = run(facts, bundles)
    assert [l.provider for l in got.links] == ["P"] and "MC" not in [e[2] for e in bound_by(one(got))]


def test_a_server_with_a_url_entry_binds_by_the_url_and_not_by_a_launch_chain():
    facts, bundles = cypher_world()
    bundles[0].append(config_entry("U1", ".mcp.json", "mcpServers.cypher.url", repo="r", fmt="mcp-json", bind="http://localhost:9999/mcp"))
    assert make_binder(facts, bundles, repos=REPOS).bind(consumer(anchor="D", site="m1", kind="mcp_tool", key="query", server="cypher")) is None


def test_a_chain_through_an_example_mcp_file_is_possible_example_config():
    facts, bundles = cypher_world(mcp_path=".mcp.example.json", mcp_role="example")
    link = one(run(facts, bundles)[0])
    assert (link.status, link.reason) == ("possible", "example-config") and bound_by(link)[0] == ("m1", 0, "L0")


def test_a_mention_of_a_server_no_mcp_json_defines_binds_nothing():
    facts, bundles = cypher_world()
    bundles[4] = unit_bundle("prompts", repo="r", root="claude", lang="markdown",
                             records=[*sym_in("D", "prompts", path="claude/a.md"), mention("m9", "D", "mcp__unknown__query")])
    assert bound_by(one(run(facts, bundles)[0])) == []


def test_a_script_name_with_a_shell_metacharacter_is_not_followed_even_when_such_a_file_exists():
    facts, bundles = cypher_world(script="cypher-mcp/we;ird.sh", args=["-c", "exec cypher-mcp/we;ird.sh"])
    assert make_binder(facts, bundles, repos=REPOS).bind(consumer(anchor="D", site="m1", kind="mcp_tool", key="query", server="cypher")) is None


def test_two_built_units_rooted_at_the_scripts_directory_leave_the_chain_unresolved_not_guessed():
    facts, bundles = cypher_world()
    bundles.append(unit_bundle("cypher-js", repo="r", root="cypher-mcp", lang="javascript", records=sym_in("PJS", "cypher-js", path="cypher-mcp/x.js")))
    assert make_binder(facts, bundles, repos=REPOS).bind(consumer(anchor="D", site="m1", kind="mcp_tool", key="query", server="cypher")) is None


# --- AN M1 / N1: the P1-11b shape of the real `.mcp.json` command, and the ordering rules -------------------------
def test_an_m1_a_command_that_is_the_script_with_no_args_names_the_script():
    """P1-11b writes the allow-listed ``exec "$CLAUDE_PROJECT_DIR/<path>"`` as ``command = <path>``, ``args = []``."""
    facts, bundles = cypher_world(command="cypher-mcp/docker-run.sh", args=[])
    got = make_binder(facts, bundles, repos=REPOS).bind(consumer(anchor="D", site="m1", kind="mcp_tool", key="query", server="cypher"))
    assert got is not None and got.units == ("cypher",) and [e.entry_id for e in got.entries] == ["L0", "LS", "LD"]


def test_an_n1_the_first_launch_unit_found_at_the_script_wins_in_id_order_not_declaration_order():
    facts, bundles = cypher_world()
    bundles[1].append(launch_fact("LA", "cypher-mcp/docker-run.sh", "command", command="sh", args=["x"]))
    bundles.append(unit_bundle("other", repo="r", root="other", records=sym_in("PO", "other")))
    facts.launch_facts.extend([launch_unit("LS", "other"), launch_unit("LA", "cypher")])        # LA sorts first, LS was declared first
    got = make_binder(facts, bundles, repos=REPOS).bind(consumer(anchor="D", site="m1", kind="mcp_tool", key="query", server="cypher"))
    assert got.units == ("cypher",)


def test_an_n1_a_script_file_record_of_another_repository_is_not_followed():
    facts, bundles = cypher_world(command="cypher-mcp/docker-run.sh", args=[])
    for r in bundles[1]:
        if r.get("type") == "file":
            r["repoId"] = "elsewhere"
    assert make_binder(facts, bundles, repos=REPOS).bind(consumer(anchor="D", site="m1", kind="mcp_tool", key="query", server="cypher")) is None


def test_sec_m4_a_server_command_of_the_wrong_shape_is_skipped_not_raised():
    for command, args in ((5, []), ("bash", "not-a-list"), ("bash", [7])):
        facts, bundles = cypher_world(command=command, args=args)
        assert make_binder(facts, bundles, repos=REPOS).bind(consumer(anchor="D", site="m1", kind="mcp_tool", key="query", server="cypher")) is None


def test_an_n1_the_script_resolver_refuses_an_absolute_token_by_itself_even_if_the_locator_would_accept_it(monkeypatch):
    facts, bundles = cypher_world(command="/cypher-mcp/docker-run.sh", args=[])
    b = make_binder(facts, bundles, repos=REPOS)
    monkeypatch.setattr(b, "_locate", lambda target: ("r", target.lstrip("/")))                 # a locator with no absolute-path guard of its own
    assert b._script_of(b._launch["L0"]) is None
    facts, bundles = cypher_world(command="cypher-mcp/docker-run.sh", args=[])
    b = make_binder(facts, bundles, repos=REPOS)
    assert b._script_of(b._launch["L0"]) == ("r", "cypher-mcp/docker-run.sh")


# --- m5 (architect ruling): the launch chain is the mention's own repository's .mcp.json only --------------------------------
def test_m5_a_mention_in_another_repository_is_not_bound_by_this_repositorys_mcp_json_server():
    facts, bundles = cypher_world()
    assert make_binder(facts, bundles, repos=REPOS).bind(consumer(anchor="D", site="m1", kind="mcp_tool", key="query", server="cypher")) is not None
    bundles[4][0]["unit"]["repoId"] = "other"                                                   # the prompts unit moves to another repository
    assert make_binder(facts, bundles, repos=REPOS).bind(consumer(anchor="D", site="m1", kind="mcp_tool", key="query", server="cypher")) is None


def test_m5_a_server_declared_in_a_file_that_is_not_a_mcp_json_is_not_a_launch_chain():
    facts, bundles = cypher_world(mcp_path="servers.json")
    assert make_binder(facts, bundles, repos=REPOS).bind(consumer(anchor="D", site="m1", kind="mcp_tool", key="query", server="cypher")) is None
