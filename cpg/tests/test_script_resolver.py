"""CO-2: the scripts resolver of ``docs/plans/connected-code-graph-p1-closeout.md`` §3.1 (assembler, tiers 1 and 2) and the
computed-command-word ``recv`` record (shell extractor). Synthetic trees only, through the real build, assembler and answer builder."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from script_resolver_helpers import assemble, ask_impact, edges_into, item_names  # noqa: E402

TARGET = "#!/bin/sh\necho target\n"
DECOY = "#!/bin/sh\necho decoy\n"


def test_a_literal_script_path_makes_the_caller_a_one_hop_caller_item_of_the_target_not_the_decoy(tmp_path):
    w = assemble(tmp_path, {"run.sh": "#!/bin/sh\n./scripts/target.sh\n", "scripts/target.sh": TARGET, "other/decoy.sh": DECOY},
                 langs=("shell",))
    doc = ask_impact(w, "scripts/target.sh")
    assert item_names(doc) == ["run.sh::<script>"]
    (item,) = doc["items"]
    assert [(h["kind"], h.get("edgeType"), h.get("dir")) for h in item["chain"]] == [
        ("Symbol", None, None), ("CallSite", "CALLS", "<-"), ("Symbol", "HAS_CALL", "<-")]       # the one-hop caller: CALLS <- then the owner
    assert (item["status"], item["distance"], item["directness"]) == ("possible", 1, "direct")
    assert item["chain"][1]["calleeName"] == "./scripts/target.sh"
    assert item_names(ask_impact(w, "other/decoy.sh")) == []


# --- tier 1: an exact path --------------------------------------------------------------------------------------
from script_resolver_helpers import edge_from  # noqa: E402


def test_tier_1_resolves_relative_to_the_calling_scripts_directory_before_the_repository_root(tmp_path):
    w = assemble(tmp_path, {"a/run.sh": "./sub/t.sh\n", "a/sub/t.sh": TARGET, "sub/t.sh": DECOY}, langs=("shell",))
    assert [e["targetPath"] for e in edge_from(w, "a/run.sh", "./sub/t.sh")] == ["a/sub/t.sh"]
    assert edges_into(w, "sub/t.sh") == []


def test_tier_1_falls_back_to_the_repository_root(tmp_path):
    w = assemble(tmp_path, {"a/run.sh": "./sub/t.sh\n", "sub/t.sh": TARGET}, langs=("shell",))
    assert [e["targetPath"] for e in edge_from(w, "a/run.sh", "./sub/t.sh")] == ["sub/t.sh"]


def test_a_tier_1_edge_takes_the_call_sites_own_status_and_reason(tmp_path):
    w = assemble(tmp_path, {"run.sh": "./scripts/t.sh\n", "scripts/t.sh": TARGET}, langs=("shell",))
    (e,) = edge_from(w, "run.sh", "./scripts/t.sh")
    assert (e["status"], e["reason"]) == ("possible", "name-match")


def test_a_launcher_argument_is_resolved_and_a_flag_before_it_is_skipped(tmp_path):
    w = assemble(tmp_path, {"run.sh": "bash ./scripts/t.sh\nbash -x ./scripts/u.sh\nexec ./scripts/v.sh\n. ./scripts/w.sh\n",
                            "scripts/t.sh": TARGET, "scripts/u.sh": TARGET, "scripts/v.sh": TARGET, "scripts/w.sh": TARGET},
                 langs=("shell",))
    got = sorted((e["targetPath"]) for callee in ("bash", "exec", ".") for e in edge_from(w, "run.sh", callee))
    assert got == ["scripts/t.sh", "scripts/u.sh", "scripts/v.sh", "scripts/w.sh"]


def test_a_command_line_string_given_to_bash_c_names_no_script(tmp_path):
    w = assemble(tmp_path, {"run.sh": "bash -c './scripts/t.sh'\n", "scripts/t.sh": TARGET}, langs=("shell",))
    assert edges_into(w, "scripts/t.sh") == []


def test_a_missing_file_gives_no_edge_and_is_counted(tmp_path):
    w = assemble(tmp_path, {"run.sh": "./scripts/gone.sh\n", "scripts/t.sh": TARGET}, langs=("shell",))
    assert edges_into(w, "scripts/t.sh") == []
    assert w.rows.dropped.get("script-call-unmatched") == 1


def test_a_script_does_not_call_itself(tmp_path):
    w = assemble(tmp_path, {"scripts/t.sh": "./scripts/t.sh\n"}, langs=("shell",))
    assert edges_into(w, "scripts/t.sh") == []


# --- tier 2: a unique path-suffix, two segments at least, `possible` ---------------------------------------------
def test_a_computed_command_word_with_a_dynamic_head_resolves_by_its_two_segment_tail_possible(tmp_path):
    w = assemble(tmp_path, {"bin/run.sh": '"$HERE/scripts/t.sh" a\n', "tools/scripts/t.sh": TARGET, "other/t.sh": DECOY}, langs=("shell",))
    (e,) = edge_from(w, "bin/run.sh", "<dynamic>")
    assert (e["targetPath"], e["status"], e["reason"]) == ("tools/scripts/t.sh", "possible", "shell-dynamic")
    doc = ask_impact(w, "tools/scripts/t.sh")
    assert item_names(doc) == ["bin/run.sh::<script>"] and doc["items"][0]["status"] == "possible"
    assert item_names(ask_impact(w, "other/t.sh")) == []


def test_a_dirname_head_with_up_segments_is_normalised_before_the_suffix_match(tmp_path):
    w = assemble(tmp_path, {"claude/a/hooks/h.sh": 'exec "$(dirname "$0")/../../scripts/guard.sh" a\n', "claude/scripts/guard.sh": TARGET,
                            "other/guard.sh": DECOY}, langs=("shell",))
    assert [e["targetPath"] for e in edge_from(w, "claude/a/hooks/h.sh", "exec")] == ["claude/scripts/guard.sh"]


def test_a_sourced_argument_with_a_dynamic_head_resolves_by_its_tail(tmp_path):
    w = assemble(tmp_path, {"bin/start.sh": '. "$(dirname "$0")/shlib/cfg.sh"\n', "bin/shlib/cfg.sh": TARGET}, langs=("shell",))
    (e,) = edge_from(w, "bin/start.sh", ".")
    assert (e["targetPath"], e["status"], e["reason"]) == ("bin/shlib/cfg.sh", "possible", "shell-dynamic")


def test_a_one_segment_tail_never_matches_even_when_one_file_has_that_basename(tmp_path):
    w = assemble(tmp_path, {"bin/run.sh": '"$HERE/t.sh" a\n. "$HERE/u.sh"\n', "scripts/t.sh": TARGET, "scripts/u.sh": TARGET}, langs=("shell",))
    assert edges_into(w, "scripts/t.sh") == [] and edges_into(w, "scripts/u.sh") == []
    assert w.rows.dropped.get("script-call-unmatched") == 2


def test_an_ambiguous_suffix_gives_no_edge_and_is_counted(tmp_path):
    w = assemble(tmp_path, {"bin/run.sh": '"$HERE/scripts/t.sh" a\n', "a/scripts/t.sh": TARGET, "b/scripts/t.sh": TARGET}, langs=("shell",))
    assert edges_into(w, "a/scripts/t.sh") == [] and edges_into(w, "b/scripts/t.sh") == []
    assert w.rows.dropped.get("script-call-ambiguous") == 1


def test_the_suffix_match_is_on_whole_segments(tmp_path):
    w = assemble(tmp_path, {"bin/run.sh": '"$HERE/scripts/t.sh" a\n', "xscripts/t.sh": TARGET}, langs=("shell",))
    assert edges_into(w, "xscripts/t.sh") == []


def test_a_decoy_with_the_same_basename_in_another_directory_is_not_a_match(tmp_path):
    w = assemble(tmp_path, {"bin/run.sh": '"$HERE/scripts/t.sh" a\n', "tools/scripts/t.sh": TARGET, "lib/t.sh": DECOY, "t.sh": DECOY},
                 langs=("shell",))
    assert [e["targetPath"] for e in edge_from(w, "bin/run.sh", "<dynamic>")] == ["tools/scripts/t.sh"]


def test_a_script_of_another_repository_is_not_a_match(tmp_path):
    w = assemble(tmp_path, {"r": {"bin/run.sh": '"$HERE/scripts/t.sh" a\n', "scripts/u.sh": TARGET}, "s": {"scripts/t.sh": TARGET}},
                 langs=("shell",))
    assert edges_into(w, "scripts/t.sh", "s") == []
    assert w.rows.dropped.get("script-call-unmatched") == 1


# --- status: a tier-1 edge carries the call site's own status; no tier-2/3 edge is ever `confirmed` ---------------
# CO-2c m2: the plan says "`confirmed` at tier 1", but the shell extractor marks every unresolved site `possible`, so real data never yields it.
# `_confirmed` below is a fabricated input (a hand-rewritten bundle the extractors never emit); the `plain` assertion is the real-data fact.
def _confirmed(records):
    return [{**r, "status": "confirmed", "reason": "resolved-call"} if r["type"] == "callsite" and not r["targets"] else r for r in records]


def test_a_tier_1_edge_is_confirmed_only_when_the_call_site_is_and_a_tier_2_edge_never_is(tmp_path):
    files = {"bin/run.sh": './scripts/a.sh\n"$HERE/scripts/b.sh"\n', "bin/scripts/a.sh": TARGET, "tools/scripts/b.sh": TARGET}
    w = assemble(tmp_path / "p", files, langs=("shell",), tweak=_confirmed)
    (t1,) = edge_from(w, "bin/run.sh", "./scripts/a.sh")
    (t2,) = edge_from(w, "bin/run.sh", "<dynamic>")
    assert (t1["targetPath"], t1["status"], t1["reason"]) == ("bin/scripts/a.sh", "confirmed", "resolved-call")
    assert t2["targetPath"] == "tools/scripts/b.sh" and t2["status"] == "possible"
    plain = assemble(tmp_path / "q", files, langs=("shell",))
    assert {e["status"] for e in calls_edges_all(plain)} == {"possible"}


def calls_edges_all(w):
    from script_resolver_helpers import calls_edges
    return calls_edges(w)


# --- Markdown mentions ------------------------------------------------------------------------------------------
def test_a_markdown_mention_of_a_script_path_is_a_possible_prompt_mention_caller(tmp_path):
    w = assemble(tmp_path, {"AGENTS.md": "# Guide\n\nRun `scripts/t.sh` first, then `t.sh` and `git`.\n", "tools/scripts/t.sh": TARGET,
                            "other/t.sh": DECOY})
    (e,) = edge_from(w, "AGENTS.md", "scripts/t.sh")
    assert (e["targetPath"], e["status"], e["reason"]) == ("tools/scripts/t.sh", "possible", "prompt-mention")
    assert edge_from(w, "AGENTS.md", "t.sh") == []                      # a bare basename is not matched
    doc = ask_impact(w, "tools/scripts/t.sh")
    assert item_names(doc) == ["AGENTS.md::<document>"] and doc["items"][0]["reason"] == "prompt-mention"


def test_a_markdown_one_segment_path_names_no_script(tmp_path):
    w = assemble(tmp_path, {"AGENTS.md": "# G\n\nSee `./t.sh`.\n", "scripts/t.sh": TARGET})
    assert edges_into(w, "scripts/t.sh") == []


def test_an_absolute_path_is_not_matched_by_its_tail_even_when_a_script_ends_the_same_way(tmp_path):
    w = assemble(tmp_path, {"run.sh": "/opt/app/scripts/u.sh\nsource /opt/app/scripts/u.sh\n", "opt/app/scripts/u.sh": TARGET}, langs=("shell",))
    assert edges_into(w, "opt/app/scripts/u.sh") == []


# --- CO-2b: the own-directory tier (closeout plan §3.5 / Q8 ruling): a one-segment tail under a computed head, matched only
# --- against the scripts in the caller's own directory, `possible`
def test_a_computed_command_word_with_a_one_segment_tail_resolves_to_the_sibling_script_possible(tmp_path):
    w = assemble(tmp_path, {"bin/run.sh": '"$HERE/t.sh" a\n', "bin/t.sh": TARGET, "scripts/t.sh": DECOY, "t.sh": DECOY}, langs=("shell",))
    (e,) = edge_from(w, "bin/run.sh", "<dynamic>")
    assert (e["targetPath"], e["status"], e["reason"]) == ("bin/t.sh", "possible", "shell-dynamic")
    doc = ask_impact(w, "bin/t.sh")
    assert item_names(doc) == ["bin/run.sh::<script>"] and doc["items"][0]["status"] == "possible"
    assert edges_into(w, "scripts/t.sh") == [] and edges_into(w, "t.sh") == []


def test_a_sourced_one_segment_argument_with_a_computed_head_resolves_to_the_sibling_script(tmp_path):
    w = assemble(tmp_path, {"bin/start.sh": '. "$(dirname "$0")/lib.sh"\nsource "$HERE/other.sh"\n', "bin/lib.sh": TARGET, "bin/other.sh": TARGET},
                 langs=("shell",))
    assert [e["targetPath"] for e in edge_from(w, "bin/start.sh", ".")] == ["bin/lib.sh"]
    assert [e["targetPath"] for e in edge_from(w, "bin/start.sh", "source")] == ["bin/other.sh"]
    assert {e["status"] for e in edge_from(w, "bin/start.sh", ".") + edge_from(w, "bin/start.sh", "source")} == {"possible"}


def test_the_own_directory_tier_never_looks_outside_the_callers_directory(tmp_path):
    w = assemble(tmp_path, {"bin/run.sh": '"$HERE/t.sh" a\n', "scripts/t.sh": TARGET, "bin/sub/t.sh": DECOY}, langs=("shell",))
    assert edges_into(w, "scripts/t.sh") == [] and edges_into(w, "bin/sub/t.sh") == []
    assert w.rows.dropped.get("script-call-unmatched") == 1


def test_a_parent_or_dot_dot_segment_under_the_head_is_not_the_own_directory(tmp_path):
    w = assemble(tmp_path, {"bin/run.sh": '"$HERE/../t.sh" a\n', "t.sh": TARGET, "bin/t.sh": DECOY}, langs=("shell",))
    assert edges_into(w, "t.sh") == [] and edges_into(w, "bin/t.sh") == []


def test_a_two_segment_tail_keeps_the_suffix_tier_and_is_not_restricted_to_the_callers_directory(tmp_path):
    w = assemble(tmp_path, {"bin/run.sh": '"$HERE/lib/t.sh" a\n', "tools/lib/t.sh": TARGET}, langs=("shell",))
    assert [e["targetPath"] for e in edge_from(w, "bin/run.sh", "<dynamic>")] == ["tools/lib/t.sh"]


def test_the_own_directory_tier_works_for_a_script_at_the_repository_root_and_not_for_itself(tmp_path):
    w = assemble(tmp_path, {"run.sh": '"$HERE/t.sh" a\n"$HERE/run.sh"\n', "t.sh": TARGET}, langs=("shell",))
    assert [e["targetPath"] for e in edge_from(w, "run.sh", "<dynamic>")] == ["t.sh"]
    assert edges_into(w, "run.sh") == []


def test_a_markdown_mention_and_a_literal_one_segment_path_are_not_the_own_directory_tier(tmp_path):
    w = assemble(tmp_path, {"bin/run.sh": "x\n", "bin/t.sh": TARGET, "AGENTS.md": "# G\n\nRun `t.sh` or `bin/t.sh`.\n"})
    assert edge_from(w, "AGENTS.md", "t.sh") == []


# --- CO-2c: the gate's findings ---------------------------------------------------------------------------------
# M1: a literal shell path that misses tier 1 gives no edge (tier 2 is for a computed head and for Markdown), a decoy sharing its tail included
def test_a_missing_literal_path_does_not_match_a_decoy_sharing_its_tail(tmp_path):
    w = assemble(tmp_path, {"run.sh": "./scripts/gone.sh\n", "lib/scripts/gone.sh": TARGET}, langs=("shell",))
    assert edges_into(w, "lib/scripts/gone.sh") == []
    assert w.rows.dropped.get("script-call-unmatched") == 1


def test_a_literal_path_climbing_out_of_the_repository_does_not_match_an_in_repository_tail(tmp_path):
    w = assemble(tmp_path, {"a/b/run.sh": "../../../x/t.sh\nbash ../../../y/t.sh\n", "x/t.sh": TARGET, "y/t.sh": TARGET, "lib/y/t.sh": TARGET},
                 langs=("shell",))
    assert edges_into(w, "x/t.sh") == [] and edges_into(w, "y/t.sh") == [] and edges_into(w, "lib/y/t.sh") == []
    assert w.rows.dropped.get("script-call-unmatched") == 2


def test_a_markdown_mention_still_matches_by_suffix_after_the_literal_restriction(tmp_path):
    w = assemble(tmp_path, {"AGENTS.md": "# G\n\nSee `scripts/gone.sh`.\n", "lib/scripts/gone.sh": TARGET})
    assert [e["targetPath"] for e in edge_from(w, "AGENTS.md", "scripts/gone.sh")] == ["lib/scripts/gone.sh"]


def test_a_markdown_absolute_path_is_not_matched_by_its_tail_even_when_a_script_ends_the_same_way(tmp_path):
    """CO-2 Pass 2 P2-2: no edge for an absolute Markdown token. Through the real pipeline the Markdown extractor already refuses it (its
    `_SCRIPT` pattern has no leading `/`), so the resolver's own refusal (`rows.py`, `text.startswith("/") and not dynamic`) is a second line
    not reachable from Markdown: removing it alone leaves this test green (mutant checked); the outcome is what is pinned."""
    w = assemble(tmp_path, {"AGENTS.md": "# G\n\nRun `/opt/app/scripts/u.sh` first.\n", "opt/app/scripts/u.sh": TARGET})
    assert edges_into(w, "opt/app/scripts/u.sh") == []
    assert edge_from(w, "AGENTS.md", "/opt/app/scripts/u.sh") == []


def test_a_markdown_path_climbing_out_of_the_repository_still_matches_its_in_repository_tail_by_design(tmp_path):
    """The prose-mention tier drops the leading `..` and suffix-matches (`possible`/`prompt-mention`); a shell literal does not (above)."""
    w = assemble(tmp_path, {"a/b/AGENTS.md": "# G\n\nRun `../../../x/t.sh`.\n", "x/t.sh": TARGET, "y/other.sh": DECOY})
    (e,) = edge_from(w, "a/b/AGENTS.md", "../../../x/t.sh")
    assert (e["targetPath"], e["status"], e["reason"]) == ("x/t.sh", "possible", "prompt-mention")
    assert edges_into(w, "y/other.sh") == []


# M2: the guards the gate's mutants found untested
def test_a_suffix_tail_names_a_script_at_the_repository_root(tmp_path):
    w = assemble(tmp_path, {"bin/run.sh": '"$HERE/scripts/t.sh" a\n', "scripts/t.sh": TARGET}, langs=("shell",))
    assert [e["targetPath"] for e in edge_from(w, "bin/run.sh", "<dynamic>")] == ["scripts/t.sh"]


def test_a_literal_path_is_never_resolved_into_another_repository(tmp_path):
    w = assemble(tmp_path, {"r": {"run.sh": "./scripts/t.sh\n", "scripts/u.sh": TARGET}, "s": {"scripts/t.sh": TARGET}}, langs=("shell",))
    assert edges_into(w, "scripts/t.sh", "s") == []


def test_zsh_is_a_launcher(tmp_path):
    w = assemble(tmp_path, {"run.sh": "zsh ./scripts/t.sh\n", "scripts/t.sh": TARGET}, langs=("shell",))
    assert [e["targetPath"] for e in edge_from(w, "run.sh", "zsh")] == ["scripts/t.sh"]


def test_a_real_segment_before_dot_dot_is_popped_in_the_tail(tmp_path):
    w = assemble(tmp_path, {"bin/run.sh": '"$HERE/a/../scripts/t.sh" a\n', "tools/scripts/t.sh": TARGET}, langs=("shell",))
    assert [e["targetPath"] for e in edge_from(w, "bin/run.sh", "<dynamic>")] == ["tools/scripts/t.sh"]


def test_only_a_shell_site_names_a_script_by_its_arguments():
    from ccg.assemble.rows import _site_path
    site = {"id": "x", "calleeName": "exec", "callKind": "command"}
    vals = {("x", "a0"): [{"const": "./scripts/t.sh"}]}
    assert _site_path(site, "shell", vals) == (False, "./scripts/t.sh")
    assert _site_path(site, "python", vals) is None and _site_path(site, "javascript", vals) is None


# m1: a call from inside a function of the script names the script itself
def test_a_call_inside_a_function_of_the_same_script_is_not_a_self_edge(tmp_path):
    w = assemble(tmp_path, {"scripts/t.sh": 'f() {\n  ./scripts/t.sh\n  "$HERE/scripts/t.sh"\n}\n', "scripts/u.sh": TARGET}, langs=("shell",))
    assert edges_into(w, "scripts/t.sh") == []


def test_a_function_of_one_script_still_calls_another_script(tmp_path):
    w = assemble(tmp_path, {"scripts/t.sh": "f() {\n  ./scripts/u.sh\n}\n", "scripts/u.sh": TARGET}, langs=("shell",))
    assert len(edges_into(w, "scripts/u.sh")) == 1


# m3: a launcher flag's value is not the script
def test_a_launcher_option_value_is_skipped_and_a_plus_flag_is_a_flag(tmp_path):
    w = assemble(tmp_path, {"run.sh": "bash --rcfile ./cfg/rc.sh ./real/a.sh\nbash -o pipefail ./real/b.sh\nbash +x ./real/c.sh\n"
                                      "bash +o nounset ./real/d.sh\nbash -O extglob ./real/e.sh\nbash --init-file ./cfg/rc.sh ./real/f.sh\n",
                            **{f"real/{c}.sh": TARGET for c in "abcdef"}, "cfg/rc.sh": TARGET}, langs=("shell",))
    assert sorted(e["targetPath"] for e in edge_from(w, "run.sh", "bash")) == [f"real/{c}.sh" for c in "abcdef"]
    assert edges_into(w, "cfg/rc.sh") == []


def test_a_combined_short_flag_holding_c_takes_a_command_line_not_a_script(tmp_path):
    w = assemble(tmp_path, {"run.sh": "bash -xc './real/a.sh'\n", "real/a.sh": TARGET}, langs=("shell",))
    assert edges_into(w, "real/a.sh") == []


# m4: a computed piece glued to the first segment makes that segment unknown
def test_a_computed_piece_glued_to_the_first_segment_matches_nothing(tmp_path):
    w = assemble(tmp_path, {"bin/run.sh": '"${D}foo/scripts/t.sh" a\n', "foo/scripts/t.sh": TARGET}, langs=("shell",))
    assert edges_into(w, "foo/scripts/t.sh") == []
    assert w.rows.dropped.get("script-call-unmatched") == 1


# m5 (resolver side): a call name the extractor kept with a `;` is refused, not matched
def test_a_callee_the_extractor_kept_with_a_semicolon_names_no_script(tmp_path):
    w = assemble(tmp_path, {"run.sh": 'x="${v:+$v; }"\necho \'${\'\n./real/a.sh; ./real/b.sh\n', "real/a.sh": TARGET, "real/b.sh": TARGET},
                 langs=("shell",))
    assert edges_into(w, "real/a.sh") == []


def test_an_own_directory_edge_is_never_confirmed_even_when_the_call_site_says_so(tmp_path):
    w = assemble(tmp_path, {"bin/run.sh": '"$HERE/t.sh" a\n', "bin/t.sh": TARGET}, langs=("shell",), tweak=_confirmed)
    (e,) = edge_from(w, "bin/run.sh", "<dynamic>")
    assert (e["targetPath"], e["status"]) == ("bin/t.sh", "possible")
