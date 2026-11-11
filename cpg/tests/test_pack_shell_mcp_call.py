"""P1-46: the shell `mcp-call` consumer pack (`cpg/packs/shell-mcp-call/`), run by the P1-16 pack runtime.

Three kinds of evidence:

* hand-built records (`packrt_helpers`) pin each clause of the rules one at a time;
* the REAL shell extractor (P1-10) turns script text into records, so what the pack matches is what an extractor emits (a
  comment makes no call site, `echo "mcp-call x"` is an `echo` call);
* `behaviour_failures(pack)` is the whole behavioural contract as one function that returns what a pack gets wrong. The
  leaf-mutation registration in `test_rule_pack_leaf_mutation.py` mutates every leaf of the pack file and requires this function
  to fail for each.

The one fixture row this pack cannot yet derive is LNK-20's quoted form `eval "mcp-call $TOOL"` (one template argument): see
`test_the_fixture_dyn_script_is_lnk_20_...` (via the P1-10b inner call site).
"""
from __future__ import annotations

import copy
import pathlib
import re
import tempfile

import pytest
import yaml

from packrt_helpers import CPG, PACKS, call, file_, pack, sym, vs
from ccg.packrt import run_packs
from ccg.rulepack import PackError, load_pack, pack_errors, validate_pack
from extractors.shell import extract_shell

PID = "shell-mcp-call"
NEGATIVES = PACKS / PID / "negatives.yaml"
FIXTURE = CPG / "fixtures" / "system1" / "fx-svc"
HISTORY = lambda path, spans: ("2024-01-01T00:00:00Z", "a" * 40)   # noqa: E731


def real_records(script: str, *, name: str = "a.sh") -> list[dict]:
    """The records the shell extractor writes for ``script`` (parsed, never run)."""
    with tempfile.TemporaryDirectory(dir=CPG / ".ccg-work" if (CPG / ".ccg-work").is_dir() else None) as d:
        (pathlib.Path(d) / name).write_text(script)
        return extract_shell(pathlib.Path(d), [name], repo_id="r", unit_id="u", history=HISTORY, warnings=[])


def consumers(records: list[dict], the_pack: dict | None = None):
    return run_packs([records], [the_pack or pack(PID)]).consumers


def view(c: dict) -> tuple:
    return (c["kind"], c["key"], c["status"], c["reason"], c["qname"])


def planted(callee: str, a0, *, kind: str = "command", line: int = 4, a1=None) -> list[dict]:
    """One script file, its ``<script>`` symbol and one call site of ``callee`` with argument ``a0`` (a const, or a dict element)."""
    recs = [file_("f1", "s.sh", lang="shell", role="primary"),
            sym("sc", "<script>", fid="f1", path="s.sh", kind="script"),
            call("c1", "sc", callee, line=line, kind=kind)]
    if a0 is not None:
        recs.append(vs("c1", "a0", a0))
    if a1 is not None:
        recs.append(vs("c1", "a1", a1))
    return recs


# ---------------------------------------------------------------------------------------------
# the pack file
# ---------------------------------------------------------------------------------------------


def test_the_pack_loads_and_passes_the_validator_and_is_a_paper_status_v1_pack():
    p = load_pack(PACKS / PID / "pack.yaml")
    assert pack_errors(p, dirname=PID) == []
    assert (p["id"], p["status"], p["packFormat"]) == (PID, "paper", 1)


# ---------------------------------------------------------------------------------------------
# the literal / dynamic split (rule `call`)
# ---------------------------------------------------------------------------------------------


def test_a_literal_tool_gives_a_confirmed_key_match_consumer_anchored_at_the_script():
    got = consumers(planted("mcp-call", "reindex"))
    assert [view(c) for c in got] == [("mcp_tool", "reindex", "confirmed", "key-match", "s.sh::<script>")]
    assert (got[0]["packId"], got[0]["ruleId"], got[0]["line"]) == (PID, "call", 4)


def test_the_tool_is_argument_a0_not_a_later_argument():
    got = consumers(planted("mcp-call", "reindex", a1="other"))
    assert [c["key"] for c in got] == ["reindex"]


def test_an_unknown_tool_value_gives_a_possible_runtime_built_consumer_with_no_key():
    got = consumers(planted("mcp-call", {"unknown": "point-derived"}))
    assert [view(c)[1:4] for c in got] == [(None, "possible", "runtime-built")]


def test_a_value_set_with_a_literal_and_an_unknown_alternative_gives_both_a_confirmed_and_a_possible_consumer():
    recs = [r for r in planted("mcp-call", None)] + [vs("c1", "a0", "reindex", {"unknown": "unresolved-ref"})]
    got = consumers(recs)
    assert sorted((c["key"] or "", c["status"]) for c in got) == [("", "possible"), ("reindex", "confirmed")]


def test_a_call_with_no_argument_gives_no_consumer():
    assert consumers(planted("mcp-call", None)) == []


def test_only_a_command_call_kind_matches_a_callee_named_mcp_call_in_another_language_does_not():
    assert consumers(planted("mcp-call", "reindex", kind="call")) == []
    assert consumers(planted("mcp-call", "reindex", kind="mention")) == []


@pytest.mark.parametrize("callee", ["mcp-calls", "xmcp-call", "mcp_call", "MCP-CALL", "./mcp-call", "mcp-call "])
def test_the_command_name_is_matched_exactly(callee):
    assert consumers(planted(callee, "reindex")) == []


# ---------------------------------------------------------------------------------------------
# under eval (rule `eval-call`)
# ---------------------------------------------------------------------------------------------


def test_a_literal_tool_under_eval_is_possible_because_the_command_is_built_at_run_time():
    got = consumers(planted("eval", "mcp-call", a1="reindex"))
    assert [view(c) for c in got] == [("mcp_tool", "reindex", "possible", "runtime-built", "s.sh::<script>")]
    assert got[0]["ruleId"] == "eval-call"


def test_a_variable_tool_under_eval_is_possible_with_no_key():
    got = consumers(planted("eval", "mcp-call", a1={"unknown": "point-derived"}))
    assert [view(c)[1:4] for c in got] == [(None, "possible", "runtime-built")]


def test_eval_of_another_command_is_not_a_consumer():
    assert consumers(planted("eval", "echo", a1="reindex")) == []
    assert consumers(planted("eval", {"unknown": "point-derived"}, a1="reindex")) == []


def test_eval_whose_first_word_only_contains_mcp_call_is_not_a_consumer():
    assert consumers(planted("eval", "mcp-calls", a1="reindex")) == []
    assert consumers(planted("eval", "x-mcp-call", a1="reindex")) == []
    assert consumers(planted("eval", "MCP-CALL", a1="reindex")) == []            # shell command names are case-sensitive


def test_a_direct_mcp_call_is_never_read_by_the_eval_rule():
    got = consumers(planted("mcp-call", "mcp-call", a1="reindex"))
    assert [(c["key"], c["ruleId"]) for c in got] == [("mcp-call", "call")]


def test_eval_with_no_tool_argument_gives_no_consumer():
    assert consumers(planted("eval", "mcp-call")) == []


def test_eval_is_matched_only_as_a_command():
    assert consumers(planted("eval", "mcp-call", kind="call", a1="reindex")) == []


# ---------------------------------------------------------------------------------------------
# the real extractor feeding the pack
# ---------------------------------------------------------------------------------------------


def test_the_fixture_refresh_script_is_lnk_19_a_confirmed_key_match_reindex_consumer_at_its_script_symbol():
    """`cpg/fixtures/system1/expected/fx-svc.yaml` LNK-19: consumer `scripts/refresh.sh::<script>`, key `reindex`."""
    recs = extract_shell(FIXTURE, ["scripts/refresh.sh"], repo_id="fx-svc", unit_id="svc", history=HISTORY, warnings=[])
    assert [view(c) for c in consumers(recs)] == [("mcp_tool", "reindex", "confirmed", "key-match", "scripts/refresh.sh::<script>")]


def test_a_script_variable_assigned_once_is_resolved_by_the_extractor_so_the_key_is_literal_and_confirmed():
    assert [view(c)[1:4] for c in consumers(real_records('T=reindex\nmcp-call "$T"\n'))] == [("reindex", "confirmed", "key-match")]


def test_a_positional_parameter_tool_is_possible_runtime_built():
    assert [view(c)[1:4] for c in consumers(real_records('mcp-call "$1" --json\n'))] == [(None, "possible", "runtime-built")]


def test_a_default_expansion_gives_the_literal_default_and_a_possible_unknown_alternative():
    got = consumers(real_records("mcp-call ${TOOL:-reindex}\n"))
    assert sorted((c["key"] or "", c["status"]) for c in got) == [("", "possible"), ("reindex", "confirmed")]


def test_a_call_inside_a_function_is_anchored_at_the_function():
    got = consumers(real_records("f() {\n  mcp-call reindex\n}\nf\n"))
    assert [(c["key"], c["qname"]) for c in got] == [("reindex", "a.sh::f")]


def test_unquoted_eval_of_mcp_call_is_possible_with_the_literal_key():
    got = consumers(real_records("eval mcp-call reindex\n"))
    assert [view(c)[1:4] for c in got] == [("reindex", "possible", "runtime-built")]


def test_unquoted_eval_of_mcp_call_with_a_variable_tool_has_no_key():
    got = consumers(real_records('eval mcp-call "$2"\n'))
    assert [view(c)[1:4] for c in got] == [(None, "possible", "runtime-built")]


def test_known_limit_a_leading_flag_is_taken_as_the_tool_and_the_real_tool_is_lost():
    """KNOWN LIMIT (analyst F1, pinned, not endorsed): the tool is `a0` blindly (P1-46 text), so `mcp-call --json reindex` gives
    a confirmed `key-match` consumer with key `--json` and no `reindex` edge. Goes red, deliberately, when the format learns to
    skip option words; the architect rules on that separately."""
    got = consumers(real_records("mcp-call --json reindex\n"))
    assert [view(c)[1:4] for c in got] == [("--json", "confirmed", "key-match")]


def test_the_fixture_dyn_script_is_lnk_20_one_possible_runtime_built_consumer_with_no_key_at_its_script_symbol():
    """LNK-20 (`eval "mcp-call $TOOL"`): P1-10b emits an inner `mcp-call` site (a0 unknown). The Link reason `shell-dynamic` is the
    linker's (P1-17 (3)); it reads the call-site shape asserted in the next test."""
    recs = extract_shell(FIXTURE, ["scripts/dyn.sh"], repo_id="fx-svc", unit_id="svc", history=HISTORY, warnings=[])
    assert [view(c) for c in consumers(recs)] == [("mcp_tool", None, "possible", "runtime-built", "scripts/dyn.sh::<script>")]
    assert [c["ruleId"] for c in consumers(recs)] == ["call"]


def test_the_call_site_shape_the_linker_reads_for_lnk_20_is_an_inner_mcp_call_and_an_eval_both_possible_shell_dynamic():
    recs = extract_shell(FIXTURE, ["scripts/dyn.sh"], repo_id="fx-svc", unit_id="svc", history=HISTORY, warnings=[])
    sites = {r["calleeName"]: (r["status"], r["reason"]) for r in recs if r["type"] == "callsite" and r["calleeName"] in ("eval", "mcp-call")}
    assert sites == {"eval": ("possible", "shell-dynamic"), "mcp-call": ("possible", "shell-dynamic")}
    inner = next(r for r in recs if r["type"] == "callsite" and r["calleeName"] == "mcp-call")
    assert [r["values"] for r in recs if r["type"] == "valueSet" and r["callSiteId"] == inner["id"] and r["arg"] == "a0"] == [[{"unknown": "runtime-built"}]]


@pytest.mark.parametrize("script", ["eval 'mcp-call reindex'\n", 'eval "mcp-call reindex"\n'])
def test_a_quoted_single_argument_eval_of_a_literal_tool_is_a_confirmed_key_match_consumer(script):
    assert [view(c)[1:4] for c in consumers(real_records(script))] == [("reindex", "confirmed", "key-match")]


@pytest.mark.parametrize("script,rule,want", [
    ("eval mcp-call reindex\n", "eval-call", [("reindex", "possible")]),            # multi-argument: no inner site
    ("eval 'mcp-call reindex'\n", "call", [("reindex", "confirmed")]),
    ('eval "mcp-call $T"\n', "call", [(None, "possible")]),
])
def test_each_eval_form_yields_exactly_one_consumer_from_one_rule_no_double_count(script, rule, want):
    got = consumers(real_records(script))
    assert [(c["key"], c["status"]) for c in got] == want and [c["ruleId"] for c in got] == [rule]


@pytest.mark.parametrize("script", ['eval "$(ssh-agent)"\n', "eval \"$(ssh-agent -s)\"\n", "eval 'echo mcp-call reindex'\n",
                                    'eval "echo $T"\n', "eval '# mcp-call reindex'\n"])
def test_eval_strings_that_do_not_run_mcp_call_are_not_consumers(script):
    assert consumers(real_records(script)) == []


# ---------------------------------------------------------------------------------------------
# negatives.yaml: one failing-then-passing negative per rule, through the real extractor
# ---------------------------------------------------------------------------------------------


def negatives() -> dict[str, dict[str, str]]:
    return yaml.safe_load(NEGATIVES.read_text())["negatives"]


def negatives_failures(the_pack: dict) -> list[str]:
    """``rule/name`` of every negative script that the pack turns into a consumer."""
    out = []
    for rule, cases in negatives().items():
        for name, script in cases.items():
            if consumers(real_records(script), the_pack):
                out.append(f"{rule}/{name}")
    return out


def test_every_rule_of_the_pack_has_negatives_and_every_negative_names_a_rule():
    rules = {r["id"] for r in pack(PID)["rules"]}
    assert set(negatives()) == rules
    for rule, cases in negatives().items():
        assert len(cases) >= 2 and all(isinstance(s, str) and s.strip() for s in cases.values()), rule


def test_the_negatives_cover_a_comment_a_string_containing_the_word_and_mcp_call_as_an_argument_to_another_command():
    names = {n for cases in negatives().values() for n in cases}
    assert {"comment", "string-containing-the-word", "argument-of-another-command"} <= names


def test_the_shipped_pack_turns_no_negative_into_a_consumer():
    assert negatives_failures(pack(PID)) == []


def _loosened(rule_id: str, **match_edits) -> dict:
    p = copy.deepcopy(pack(PID))
    rule = next(r for r in p["rules"] if r["id"] == rule_id)
    rule["match"]["callsite"].update(match_edits)
    return p


@pytest.mark.parametrize("rule,mutant,expected", [
    # the alternative the plan argued down: match any call that mentions the word, not the command itself
    ("call", {"callee": {"glob": "*mcp-call*"}}, "call/"),
    ("call", {"callee": {"regex": ".*"}}, "call/"),
    ("eval-call", {"args": [{"arg": "a0"}]}, "eval-call/"),                      # any eval with a first word
    ("eval-call", {"args": [{"arg": "a0", "value": {"regex": ".*mcp-call.*"}}]}, "eval-call/"),   # word anywhere in a0
])
def test_a_loosened_rule_turns_a_negative_of_that_rule_into_a_consumer(rule, mutant, expected):
    got = negatives_failures(_loosened(rule, **mutant))
    assert any(g.startswith(expected) for g in got), (mutant, got)


# ---------------------------------------------------------------------------------------------
# the contract as one function (used by the leaf-mutation registration)
# ---------------------------------------------------------------------------------------------


def behaviour_failures(the_pack: dict) -> list[str]:
    """What ``the_pack`` gets wrong, as short strings (empty for the shipped pack). The pack is run through the validator, the
    positive and negative cases above in one place, so a mutated leaf has to be caught by something here."""
    bad: list[str] = []
    errs = pack_errors(the_pack, dirname=PID)
    if errs:
        bad.append(f"invalid: {errs[0]}")
        return bad

    def got(recs):
        return [view(c)[1:4] + (c["packId"], c["ruleId"]) for c in run_packs([recs], [the_pack]).consumers]

    cases = [
        (planted("mcp-call", "reindex"), [("reindex", "confirmed", "key-match", PID, "call")]),
        (planted("mcp-call", {"unknown": "u"}), [(None, "possible", "runtime-built", PID, "call")]),
        (planted("mcp-call", "reindex", kind="call"), []),
        (planted("mcp-calls", "reindex"), []),
        (planted("mcp-call", "reindex", a1="other"), [("reindex", "confirmed", "key-match", PID, "call")]),
        (planted("mcp-call", None), []),
        (planted("eval", "mcp-call", a1="reindex"), [("reindex", "possible", "runtime-built", PID, "eval-call")]),
        (planted("eval", "mcp-call", a1={"unknown": "u"}), [(None, "possible", "runtime-built", PID, "eval-call")]),
        (planted("eval", "echo", a1="reindex"), []),
        (planted("eval", "mcp-calls", a1="reindex"), []),
        (planted("eval", "MCP-CALL", a1="reindex"), []),
        (planted("eval", "mcp-call"), []),
        (planted("eval", "mcp-call", kind="call", a1="reindex"), []),
    ]
    for i, (recs, want) in enumerate(cases):
        if got(recs) != want:
            bad.append(f"case {i}: {got(recs)} != {want}")
    bad += [f"negative {n}" for n in negatives_failures(the_pack)]
    return bad


def test_the_shipped_pack_has_no_behaviour_failures():
    assert behaviour_failures(pack(PID)) == []
