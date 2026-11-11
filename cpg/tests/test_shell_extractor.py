"""P1-10 — the shell extractor (`extractors/shell/extract.py`): goldens per construct, hostile input, the corpus smoke.

Expected values are written by hand from the plan's text and the bundle spec; nothing here computes a qname or an id
except through `ccg.ids`. The bundle is validated by `ccg.bundle` (`dumps` sorts, then `validate`), and the writer
backstop `writer_rules.audit_records` runs over the records with the texts the extractor hashed (none: it writes no
`configEntry`/`launchFact`, and a `${var:-default}` default is never a `bindValue`).
"""
from __future__ import annotations

import os
import signal
import stat
from collections import Counter
from pathlib import Path

import pytest

from ccg import bundle, ids
from extractors.common import writer_rules as wr
from extractors.shell import MAX_FILE_BYTES, SourceRefused, extract_shell
from extractors.shell import extract as shell

CPG = Path(__file__).resolve().parent.parent
REPO_ROOT = CPG.parent
DATA = Path(__file__).resolve().parent / "data" / "shell"
REPO = "shellrepo"
UNIT = ids.unit_id(REPO, ".", "shell")
KEY = bytes(range(64, 96))                   # a test key; the real one is never read here
STAMP = ("2026-10-29T12:00:00Z", "1" * 40)
AUDIT = "0f8fad5b-d9cb-469f-a165-70867728950e"


def history(path, regions):
    history.calls.append((path, regions))
    return STAMP


history.calls = []


def run(files, root=DATA, **kw):
    warnings: list = []
    recs = extract_shell(root, files if isinstance(files, list) else [files], repo_id=REPO, unit_id=UNIT,
                         history=history, warnings=warnings, **kw)
    return recs, warnings


def snippet(tmp_path, text: str, name="s.sh"):
    (tmp_path / name).write_text(text, encoding="utf-8")
    return run([name], root=tmp_path)


def header(records) -> dict:
    counts = Counter(r["type"] for r in records)
    return {
        "type": "header", "bundleSchemaVersion": "1.0", "unit": {"repoId": REPO, "rootPath": ".", "lang": "shell"},
        "unitId": UNIT, "builtCommit": "1" * 40, "builtTree": "2" * 40, "builtAt": "2026-10-29T12:00:00Z",
        "tools": {"joern": "none", "frontend": "tree-sitter-bash", "frontendVersion": "0", "normalizer": "shell",
                  "normalizerVersion": "0", "extractScript": "extract.py", "extractScriptHash": "0" * 64},
        "sandboxAuditIds": [AUDIT], "egressMode": "offline", "flowProjected": False, "fieldDepthCap": 3,
        "vendoredDirs": [], "history": "available",
        "counts": {t: counts.get(t, 0) for t in bundle.RECORD_TYPES[1:]},
    }


def assert_valid(records):
    """The records are a valid bundle body (`dumps` sorts and validates) and pass the writer backstop."""
    bundle.dumps([header(records), *records])
    assert not any(r["type"] in ("configEntry", "launchFact") for r in records)
    texts = {r["id"]: "" for r in records if r["type"] in ("configEntry", "launchFact")}
    assert wr.audit_records(records, key=KEY, declared_services=set(), texts=texts) == []


# --- lookups ------------------------------------------------------------------------------------------------

def of(records, t):
    return [r for r in records if r["type"] == t]


def sym(records, qname):
    (s,) = [r for r in of(records, "symbol") if r["qname"] == qname]
    return s


def calls(records, scope_qname, callee):
    sid = sym(records, scope_qname)["id"]
    return sorted((r for r in of(records, "callsite") if r["symbolId"] == sid and r["calleeName"] == callee),
                  key=lambda r: r["ordinal"])


def call(records, scope_qname, callee, ordinal=0):
    return calls(records, scope_qname, callee)[ordinal]


def values(records, cs, arg):
    (v,) = [r["values"] for r in of(records, "valueSet") if r["callSiteId"] == cs["id"] and r["arg"] == arg]
    return v


def flows(records, cs):
    """((toPath, fromPath, defKind, status, reason), ...) of the data records whose use is `cs`."""
    pts = {r["id"]: r["type"] for r in records if r["type"] in ("param", "callsite")}
    return sorted((d["toPath"], d["fromPath"], pts[d["def"]], d["status"], d["reason"])
                  for d in of(records, "data") if d["use"] == cs["id"])


def params(records, qname):
    sid = sym(records, qname)["id"]
    return sorted((p["index"], p["name"]) for p in of(records, "param") if p["symbolId"] == sid)


# --- symbols ------------------------------------------------------------------------------------------------

def test_one_script_symbol_per_file_and_one_symbol_per_function_all_top_level(tmp_path):
    recs, w = snippet(tmp_path, "a() { :; }\nfunction b {\n  c() { :; }\n}\nfunction d() { :; }\necho hi\n")
    assert w == []
    syms = {s["qname"]: s for s in of(recs, "symbol")}
    assert sorted(syms) == ["s.sh::<script>", "s.sh::a", "s.sh::b", "s.sh::c", "s.sh::d"]   # c is top level (spec3 "Shell")
    script = syms["s.sh::<script>"]
    assert (script["kind"], script["name"], script["lineStart"], script["lineEnd"], "parentId" in script) == \
        ("script", "<script>", 1, 6, False)
    assert {syms[q]["kind"] for q in ("s.sh::a", "s.sh::b", "s.sh::c", "s.sh::d")} == {"function"}
    assert {syms[q]["parentId"] for q in ("s.sh::a", "s.sh::b", "s.sh::c", "s.sh::d")} == {script["id"]}
    assert (syms["s.sh::b"]["lineStart"], syms["s.sh::b"]["lineEnd"]) == (2, 4)
    assert syms["s.sh::a"]["id"] == ids.symbol_id("shell", REPO, "s.sh::a")
    assert_valid(recs)


def test_a_redefinition_gets_the_hash_n_suffix_and_calls_resolve_to_every_definition(tmp_path):
    recs, _ = snippet(tmp_path, "f() { :; }\nif true; then f() { :; }; fi\nf\n")
    assert sorted(s["qname"] for s in of(recs, "symbol")) == ["s.sh::<script>", "s.sh::f", "s.sh::f#2"]
    assert sym(recs, "s.sh::f#2")["name"] == "f"
    (c,) = calls(recs, "s.sh::<script>", "f")
    assert [t["qname"] for t in c["targets"]] == sorted((t["qname"] for t in c["targets"]),
                                                        key=lambda q: ids.symbol_id("shell", REPO, q))
    assert {t["qname"] for t in c["targets"]} == {"s.sh::f", "s.sh::f#2"}
    assert_valid(recs)


def test_the_file_record_and_an_empty_file(tmp_path):
    recs, w = snippet(tmp_path, "")
    (f,) = of(recs, "file")
    assert (f["kind"], f["lang"], f["role"], f["lineCount"], f["unitId"]) == ("script", "shell", "primary", 0, UNIT)
    assert f["contentHash"] == shell._sha256(b"")
    assert (sym(recs, "s.sh::<script>")["lineStart"], sym(recs, "s.sh::<script>")["lineEnd"]) == (1, 1)
    assert w == []
    assert_valid(recs)
    (tmp_path / "ex").mkdir()
    (tmp_path / "ex" / "run.sh.example").write_text("echo\n")
    assert run(["ex/run.sh.example"], root=tmp_path)[0][0]["role"] == "example"


def test_last_changed_comes_from_the_callers_history_over_the_symbols_span(tmp_path):
    history.calls.clear()
    recs, _ = snippet(tmp_path, "a() {\n  :\n}\n")
    assert history.calls == [("s.sh", [(1, 3)]), ("s.sh", [(1, 3)])]            # <script> (whole file), then a
    assert {s["lastChangedAt"] for s in of(recs, "symbol")} == {STAMP[0]}
    (tmp_path / "t.sh").write_text("a() { :; }\n")
    bare = extract_shell(tmp_path, ["t.sh"], repo_id=REPO, unit_id=UNIT)
    assert all("lastChangedAt" not in s for s in of(bare, "symbol"))


# --- parameters and flow ------------------------------------------------------------------------------------

def test_positional_parameters_are_params_and_flow_to_the_command_argument():
    recs, w = run("positional.sh")
    assert w == []
    assert params(recs, "positional.sh::first") == [(0, "$1")]
    assert params(recs, "positional.sh::pair") == [(0, "$1"), (1, "$2")]
    cp = call(recs, "positional.sh::pair", "cp")
    assert flows(recs, cp) == [("a0", "p0", "param", "confirmed", "intra-flow"),
                               ("a1", "p1", "param", "confirmed", "intra-flow")]
    assert values(recs, cp, "a0") == [{"unknown": "point-derived"}]
    echo = call(recs, "positional.sh::first", "echo")
    (d,) = [d for d in of(recs, "data") if d["use"] == echo["id"]]
    assert d["def"] == ids.point_id(sym(recs, "positional.sh::first")["id"], "param", 0)
    assert_valid(recs)


def test_only_the_second_positional_still_gives_param_zero_and_one():
    recs, _ = run("positional.sh")
    assert params(recs, "positional.sh::second_only") == [(0, "$1"), (1, "$2")]
    echo = call(recs, "positional.sh::second_only", "echo")
    assert flows(recs, echo) == [("a0", "p1", "param", "confirmed", "intra-flow")]       # ${2} -> index 1


def test_at_sign_is_one_variadic_param_after_the_numbered_ones_and_its_value_is_a_spread():
    recs, _ = run("positional.sh")
    assert params(recs, "positional.sh::rest") == [(0, "$1"), (1, "$@")]
    echo = call(recs, "positional.sh::rest", "echo")
    assert values(recs, echo, "a1") == [{"unknown": "spread-argument"}]
    assert flows(recs, echo) == [("a0", "p0", "param", "confirmed", "intra-flow"),
                                 ("a1", "p1", "param", "confirmed", "intra-flow")]
    assert params(recs, "positional.sh::only_all") == [(0, "$@")]
    assert flows(recs, call(recs, "positional.sh::only_all", "exec")) == \
        [("a0", "p0", "param", "confirmed", "intra-flow")]
    assert params(recs, "positional.sh::<script>") == [(0, "$1"), (1, "$@")]          # the script's own arguments


def test_single_quotes_and_an_escaped_dollar_are_not_parameters_and_an_interpolation_is_a_template():
    recs, _ = run("positional.sh")
    assert params(recs, "positional.sh::literals") == [(0, "$1")]                   # only "x$1y"
    echo = call(recs, "positional.sh::literals", "echo")
    assert values(recs, echo, "a0") == [{"const": "$1"}]
    assert values(recs, echo, "a1") == [{"const": "$1"}]
    assert values(recs, echo, "a2") == [{"template": [{"const": "x"}, {"unknown": "point-derived"}, {"const": "y"}]}]
    assert flows(recs, echo) == [("a2", "p0", "param", "confirmed", "intra-flow")]


def test_a_positional_beyond_the_cap_is_a_value_not_a_param(tmp_path):
    recs, _ = snippet(tmp_path, f"f() {{ echo \"${{{shell.MAX_POSITIONAL + 1}}}\" \"${{{shell.MAX_POSITIONAL}}}\"; }}\n")
    assert len(params(recs, "s.sh::f")) == shell.MAX_POSITIONAL
    echo = call(recs, "s.sh::f", "echo")
    assert values(recs, echo, "a0") == [{"unknown": "runtime-built"}]
    assert [f[1] for f in flows(recs, echo)] == [f"p{shell.MAX_POSITIONAL - 1}"]


# --- call sites ---------------------------------------------------------------------------------------------

def test_call_sites_belong_to_the_innermost_symbol_with_source_order_ordinals(tmp_path):
    recs, _ = snippet(tmp_path, "echo top\nf() {\n  echo in\n  echo $(echo nested) again\n}\necho top2\n")
    top = calls(recs, "s.sh::<script>", "echo")
    assert [(c["ordinal"], c["line"]) for c in top] == [(0, 1), (1, 6)]
    inner = calls(recs, "s.sh::f", "echo")
    assert [(c["ordinal"], c["line"]) for c in inner] == [(0, 3), (1, 4), (2, 4)]
    assert [values(recs, c, "a0") for c in inner] == [[{"const": "in"}], [{"unknown": "point-derived"}], [{"const": "nested"}]]
    assert all(c["callKind"] == "command" and c["id"] == ids.callsite_id(c["symbolId"], "echo", c["ordinal"]) for c in inner)
    assert_valid(recs)


def test_ordinals_count_per_callee_name_and_a_quoted_command_word_is_its_text(tmp_path):
    recs, _ = snippet(tmp_path, "a\nb\na\n\"b\"\nb\n'c d' x\n\"echo hi\"\n")
    sid = sym(recs, "s.sh::<script>")["id"]
    seen = sorted((c["line"], c["calleeName"], c["ordinal"]) for c in of(recs, "callsite") if c["symbolId"] == sid)
    assert seen == [(1, "a", 0), (2, "b", 0), (3, "a", 1), (4, "b", 1), (5, "b", 2),
                    (6, "<dynamic>", 0), (7, "<dynamic>", 1)]            # a word with a space is not a command name
    assert_valid(recs)


def test_a_script_variable_read_before_its_assignment_is_not_a_ref(tmp_path):
    recs, _ = snippet(tmp_path, "use \"$A\"\nA=1\nuse \"$A\"\nf() { use \"$A\"; }\n")
    first, second = calls(recs, "s.sh::<script>", "use")
    assert values(recs, first, "a0") == [{"unknown": "unresolved-ref"}]
    assert values(recs, second, "a0") == [{"const": "1"}]                  # inline, from the reaching assignment
    assert values(recs, call(recs, "s.sh::f", "use"), "a0") == [{"ref": "s.sh::A"}]


def test_the_call_site_line_is_the_commands_word_and_a_line_continuation_joins_a_string(tmp_path):
    recs, _ = snippet(tmp_path, "FOO=1 \\\n  cmd one \\\n  two \"th\\\nree\"\n")
    c = call(recs, "s.sh::<script>", "cmd")
    assert c["line"] == 2                                                   # the callee's line, not the assignment's
    assert [values(recs, c, f"a{i}") for i in range(3)] == [[{"const": x}] for x in ("one", "two", "three")]


def test_a_command_resolves_to_a_function_of_the_same_file_and_others_are_name_matches():
    recs, _ = run("gquery.sh")
    q = call(recs, "gquery.sh::bootstrap_reference", "gquery")
    assert (q["resolution"], q["status"], q["reason"]) == ("resolved", "confirmed", "resolved-call")
    assert q["targets"] == [{"symbolId": sym(recs, "gquery.sh::gquery")["id"], "qname": "gquery.sh::gquery"}]
    cli = call(recs, "gquery.sh::gquery", "redis-cli")
    assert (cli["resolution"], cli["targets"], cli["status"], cli["reason"]) == \
        ("unresolved", [], "possible", "name-match")


def test_a_function_parameter_lifts_a_literal_at_its_call_site():
    """The `gquery` of bootstrap_schema.sh: the wrapper's `$2` reaches `redis-cli`, each call site carries the Cypher."""
    recs, _ = run("gquery.sh")
    cli = call(recs, "gquery.sh::gquery", "redis-cli")
    assert flows(recs, cli) == [("a5", "p0", "param", "confirmed", "intra-flow"),
                                ("a6", "p1", "param", "confirmed", "intra-flow")]
    assert params(recs, "gquery.sh::gquery") == [(0, "$1"), (1, "$2")]
    sites = calls(recs, "gquery.sh::bootstrap_reference", "gquery")
    assert [values(recs, c, "a1") for c in sites] == [[{"const": "CREATE INDEX FOR (n:WorkflowDef) ON (n.key)"}],
                                                      [{"const": "CREATE INDEX FOR (n:Entity) ON (n.entityId)"}]]
    assert [values(recs, c, "a0") for c in sites] == [[{"const": "reference"}]] * 2        # `local g="reference"`
    top = call(recs, "gquery.sh::<script>", "gquery")
    assert values(recs, top, "a0") == [{"template": [{"const": "ws:"}, {"unknown": "point-derived"}]}]
    assert flows(recs, top) == [("a0", "p0", "param", "confirmed", "intra-flow")]            # the script's own $1
    gc = call(recs, "gquery.sh::bootstrap_reference", "gconstraint")
    assert [values(recs, gc, f"a{i}") for i in range(1, 7)] == [[{"const": x}] for x in
                                                                ("UNIQUE", "NODE", "Entity", "PROPERTIES", "1", "entityId")]
    assert_valid(recs)


def test_a_script_variable_assigned_once_is_a_constant_and_a_function_sees_it_by_ref():
    recs, _ = run("gquery.sh")
    consts = {c["qname"]: c["values"] for c in of(recs, "constant")}
    assert consts == {
        "gquery.sh::HOST": [{"const": "127.0.0.1"}, {"unknown": "unresolved-ref"}],
        "gquery.sh::PORT": [{"const": "6379"}, {"unknown": "unresolved-ref"}],
    }
    cli = call(recs, "gquery.sh::gquery", "redis-cli")
    assert values(recs, cli, "a1") == [{"ref": "gquery.sh::HOST"}]
    assert values(recs, cli, "a3") == [{"ref": "gquery.sh::PORT"}]
    (c,) = [c for c in of(recs, "constant") if c["qname"].endswith("::HOST")]
    assert c["id"] == ids.constant_id("shell", REPO, "gquery.sh::HOST") and c["line"] == 5


def test_constants_need_exactly_one_unconditional_script_level_assignment(tmp_path):
    recs, _ = snippet(tmp_path, "A=1\nA=2\nB=1\nif x; then C=1; fi\nD=1\nf() { D=2; E=3; echo \"$A$B$C$D$E\"; }\n"
                                "F=$1\nG=1\ng() { local G=2; echo \"$G\"; }\nH=1\nH() { :; }\n")
    assert [c["qname"] for c in of(recs, "constant")] == ["s.sh::B"]       # A twice, C conditional, D reassigned,
    echo = call(recs, "s.sh::f", "echo")                                   # F point-derived only, G shadowed, H a function
    assert values(recs, echo, "a0") == [{"template": [{"unknown": "unresolved-ref"}, {"ref": "s.sh::B"},
                                                      {"unknown": "unresolved-ref"}, {"const": "23"}]}]


# --- shell-dynamic ------------------------------------------------------------------------------------------

def test_eval_is_possible_shell_dynamic_and_what_flows_into_it_is_a_dynamic_flow():
    recs, _ = run("dynamic.sh")
    ev = call(recs, "dynamic.sh::run_eval", "eval")
    assert (ev["status"], ev["reason"], ev["targets"]) == ("possible", "shell-dynamic", [])
    assert flows(recs, ev) == [("a0", "p0", "param", "possible", "shell-dynamic")]


def test_a_literal_eval_is_still_shell_dynamic(tmp_path):
    recs, _ = snippet(tmp_path, "eval 'echo hi'\n")
    assert call(recs, "s.sh::<script>", "eval")["reason"] == "shell-dynamic"


def test_indirect_expansion_makes_the_call_site_and_its_flow_dynamic():
    recs, _ = run("dynamic.sh")
    echo = call(recs, "dynamic.sh::indirect", "echo")
    assert (echo["status"], echo["reason"]) == ("possible", "shell-dynamic")
    assert values(recs, echo, "a0") == [{"unknown": "runtime-built"}]
    assert flows(recs, echo) == [("a0", "p0", "param", "possible", "shell-dynamic")]       # n <- $1, through ${!n}


def test_a_computed_source_is_shell_dynamic_and_a_literal_source_is_a_name_match():
    recs, _ = run("dynamic.sh")
    for fn in ("load", "computed"):
        c = call(recs, f"dynamic.sh::{fn}", "source")
        assert (c["status"], c["reason"]) == ("possible", "shell-dynamic"), fn
    lit = call(recs, "dynamic.sh::load_lib", ".")
    assert (lit["status"], lit["reason"], values(recs, lit, "a0")) == ("possible", "name-match", [{"const": "./lib.sh"}])


def test_a_computed_command_name_is_dynamic():
    recs, _ = run("dynamic.sh")
    for scope in ("dynamic.sh::dyncmd", "dynamic.sh::<script>"):
        (c,) = calls(recs, scope, "<dynamic>")
        assert (c["status"], c["reason"]) == ("possible", "shell-dynamic")
    assert_valid(recs)


# --- quoting and values -------------------------------------------------------------------------------------

def test_a_default_expansion_is_an_alternative_and_never_a_bind_value_or_default_field():
    recs, _ = run("quoting.sh")
    echoes = calls(recs, "quoting.sh::f", "echo")
    assert values(recs, echoes[0], "a0") == [{"const": "fallback"}, {"unknown": "unresolved-ref"}]
    assert values(recs, echoes[1], "a0") == [{"const": "8200"}, {"unknown": "unresolved-ref"}]
    assert not any(k in r for r in recs for k in ("bindValue", "default", "valueHash", "defaultHash", "command", "args"))
    assert not any(r["type"] in ("configEntry", "launchFact") for r in recs)
    assert_valid(recs)


def test_quoting_edge_cases():
    recs, _ = run("quoting.sh")
    e = calls(recs, "quoting.sh::f", "echo")
    assert values(recs, e[2], "a0") == [{"const": "single $1 $(not run)"}]
    assert values(recs, e[3], "a0") == [{"const": 'esc " $HOME'}]
    assert values(recs, e[4], "a0") == [{"const": "abc"}]
    assert values(recs, e[5], "a0") == [{"template": [{"const": "pre-"}, {"unknown": "unresolved-ref"},
                                                      {"const": "-post"}]}]
    assert values(recs, e[6], "a0") == [{"unknown": "point-derived"}]                  # "$(date)"
    assert values(recs, e[7], "a0") == [{"unknown": "runtime-built"}]                  # ${v%.txt}: derived
    assert flows(recs, e[6]) == [("a0", "r", "callsite", "confirmed", "intra-flow")]
    assert len(calls(recs, "quoting.sh::f", "date")) == 1                              # the substitution is a call site
    assert params(recs, "quoting.sh::f") == []                                         # '$1' in single quotes: no param


def test_assignments_reach_their_uses_by_reaching_definition():
    recs, _ = run("flow.sh")
    S = "flow.sh::f"
    use = lambda n: call(recs, S, n)                                                    # noqa: E731
    assert values(recs, use("use_a"), "a0") == [{"unknown": "point-derived"}]
    assert flows(recs, use("use_a")) == [("a0", "p0", "param", "confirmed", "intra-flow")]       # b <- a <- $1
    assert values(recs, use("use_x"), "a0") == [{"const": "two"}]                                # a later one kills
    assert values(recs, use("use_y"), "a0") == [{"const": "left"}, {"const": "right"}]           # both branches reach
    assert values(recs, use("use_z"), "a0") == [{"unknown": "unresolved-ref"}]                   # z="$z": a self-reference


def test_a_command_substitution_flows_by_its_result_through_a_variable_and_a_pipeline():
    recs, _ = run("flow.sh")
    S = "flow.sh::f"
    date = call(recs, S, "date")
    d = call(recs, S, "use_d")
    assert flows(recs, d) == [("a0", "r", "callsite", "confirmed", "intra-flow")]
    (dd,) = [x for x in of(recs, "data") if x["use"] == d["id"]]
    assert dd["def"] == date["id"] and dd["toPath"] == "a0"
    sub = call(recs, S, "use_sub")
    (sd,) = [x for x in of(recs, "data") if x["use"] == sub["id"]]
    assert sd["def"] == call(recs, S, "inner")["id"]
    assert flows(recs, call(recs, S, "inner")) == [("a0", "p0", "param", "confirmed", "intra-flow")]
    pipe = call(recs, S, "use_pipe")
    (pd,) = [x for x in of(recs, "data") if x["use"] == pipe["id"]]
    assert pd["def"] == call(recs, S, "tr")["id"]                     # the last command of the pipeline gives the result
    assert_valid(recs)


def test_the_value_set_alternatives_are_sorted_deduplicated_and_capped(tmp_path):
    def program(n, use):
        arms = "\n".join(f"  if [ $1 = {i} ]; then v=x{i}; fi" for i in range(n))
        return f"f() {{\n  v=base\n{arms}\n  use {use}\n}}\n"
    recs, _ = snippet(tmp_path, program(14, '"$v" "$v"'))                  # 15 reach: all kept, sorted, once each
    vals = values(recs, call(recs, "s.sh::f", "use"), "a0")
    assert len(vals) == 15 and {"unknown": "value-set-cap"} not in vals
    assert [shell._canon(v) for v in vals] == sorted(shell._canon(v) for v in vals)
    assert values(recs, call(recs, "s.sh::f", "use"), "a1") == vals
    recs, _ = snippet(tmp_path, program(15, '"${v:-fallback}"'))            # 16 reach + a default: 17, trimmed to 16
    vals = values(recs, call(recs, "s.sh::f", "use"), "a0")
    assert len(vals) == shell.MAX_ALTERNATIVES and {"unknown": "value-set-cap"} in vals
    assert [shell._canon(v) for v in vals] == sorted(shell._canon(v) for v in vals)
    recs, _ = snippet(tmp_path, program(30, '"$v"'))                        # more than 16 definitions reach: no value
    assert values(recs, call(recs, "s.sh::f", "use"), "a0") == [{"unknown": "value-set-cap"}]
    assert_valid(recs)


def test_read_for_append_and_bare_declarations_define_a_variable_without_a_value(tmp_path):
    recs, _ = snippet(tmp_path, "f() {\n  read -r -p 'q: ' line rest\n  for v in a b; do use_v \"$v\"; done\n"
                                "  w=base\n  w+=more\n  use_w \"$w\"\n  local u\n  use_u \"$u\"\n  use_l \"$line\" \"$rest\"\n}\n")
    runtime = [{"unknown": "runtime-built"}]
    for name in ("use_v", "use_u"):
        assert values(recs, call(recs, "s.sh::f", name), "a0") == runtime, name
    assert values(recs, call(recs, "s.sh::f", "use_w"), "a0") == [{"const": "base"}, {"unknown": "runtime-built"}]
    use_l = call(recs, "s.sh::f", "use_l")
    assert (values(recs, use_l, "a0"), values(recs, use_l, "a1")) == (runtime, runtime)       # not `unresolved-ref`


def test_prefix_assignments_redirects_and_heredocs_are_not_arguments(tmp_path):
    recs, _ = snippet(tmp_path, "f() {\n  FOO=bar cmd \"$FOO\" two > \"$1\" 2>&1\n  cat <<EOF\nhello $(inner \"$2\")\nEOF\n}\n")
    cmd = call(recs, "s.sh::f", "cmd")
    assert [r["arg"] for r in of(recs, "valueSet") if r["callSiteId"] == cmd["id"]] == ["a0", "a1"]
    assert values(recs, cmd, "a0") == [{"unknown": "unresolved-ref"}]                         # FOO=bar is not a definition
    assert flows(recs, cmd) == []
    inner = call(recs, "s.sh::f", "inner")                                                    # a call site inside the heredoc
    assert flows(recs, inner) == [("a0", "p1", "param", "confirmed", "intra-flow")]
    assert not [r for r in of(recs, "valueSet") if r["callSiteId"] == call(recs, "s.sh::f", "cat")["id"]]


def test_a_symbols_fingerprint_is_the_hash_of_its_stripped_non_blank_lines(tmp_path):
    recs, _ = snippet(tmp_path, "#!/bin/sh\n\nf() {\n    echo   hi\n\n}\n")
    assert sym(recs, "s.sh::f")["fingerprint"] == shell._sha256(b"f() {\necho   hi\n}")
    assert sym(recs, "s.sh::<script>")["fingerprint"] == shell._sha256(b"#!/bin/sh\nf() {\necho   hi\n}")
    assert {s["complexity"] for s in of(recs, "symbol")} == {1}               # no cond records in P1: 1 + 0 (spec4 section 2)


def test_strings_are_nfc_and_odd_function_names_are_valid(tmp_path):
    recs, _ = snippet(tmp_path, "my-func.v2() { echo \"cafe\u0301\"; }\nmy-func.v2\n")
    assert values(recs, call(recs, "s.sh::my-func.v2", "echo"), "a0") == [{"const": "caf\u00e9"}]
    assert call(recs, "s.sh::<script>", "my-func.v2")["status"] == "confirmed"
    assert_valid(recs)


def test_a_parser_that_runs_out_of_time_is_a_warning_with_a_call_less_script_symbol(tmp_path, monkeypatch):
    monkeypatch.setattr(shell, "PARSE_TIMEOUT_SECONDS", -1.0)
    recs, w = snippet(tmp_path, "echo hi\n")
    assert [x["code"] for x in w] == ["parse-timeout"]
    assert [s["qname"] for s in of(recs, "symbol")] == ["s.sh::<script>"] and not of(recs, "callsite")


def test_thousands_of_definitions_of_one_variable_give_a_capped_value_not_a_quadratic_walk(tmp_path):
    recs, _ = snippet(tmp_path, "f() {\n" + "".join(f"  if t; then v={i}; fi\n" for i in range(3000)) +
                      "".join('  use "$v"\n' for _ in range(3000)) + "}\n")
    assert {tuple(map(shell._canon, values(recs, c, "a0"))) for c in calls(recs, "s.sh::f", "use")} == \
        {(shell._canon({"unknown": "value-set-cap"}),)}


# --- secrets are withheld -----------------------------------------------------------------------------------

def test_a_literal_that_fails_the_token_screen_or_follows_a_secret_named_flag_is_withheld(tmp_path):
    token = "ghp_" + "a1B2c3D4" * 5
    recs, _ = snippet(tmp_path, f"curl --password hunter2 --api-key=hunter2 -H '{token}' --host example\n")
    c = call(recs, "s.sh::<script>", "curl")
    assert [values(recs, c, f"a{i}") for i in range(7)] == [
        [{"const": "--password"}], [{"unknown": "runtime-built"}],          # the value after a secret-named flag
        [{"unknown": "runtime-built"}],                                     # --api-key=hunter2: the name is in the word
        [{"const": "-H"}], [{"unknown": "runtime-built"}],                  # a token-shaped literal
        [{"const": "--host"}], [{"const": "example"}]]
    assert "hunter2" not in repr(recs) and token not in repr(recs)


def test_a_variable_whose_name_fails_the_name_rule_never_yields_its_value(tmp_path):
    recs, _ = snippet(tmp_path, "API_TOKEN=abc123\nuse \"$API_TOKEN\" \"$OK\"\nOK=fine\nf() { use \"$API_TOKEN\"; }\n"
                                "g() { local DB_PASSWORD=hunter2; use \"$DB_PASSWORD\"; }\n")
    assert values(recs, call(recs, "s.sh::<script>", "use"), "a0") == [{"unknown": "runtime-built"}]
    assert [c["qname"] for c in of(recs, "constant")] == ["s.sh::OK"]                  # no constant for API_TOKEN
    assert values(recs, call(recs, "s.sh::f", "use"), "a0") == [{"unknown": "unresolved-ref"}]
    assert values(recs, call(recs, "s.sh::g", "use"), "a0") == [{"unknown": "runtime-built"}]
    assert "hunter2" not in repr(recs) and "abc123" not in repr(recs)


# --- unparseable, hostile and bounded input ------------------------------------------------------------------

def test_an_unparseable_file_is_a_warning_and_keeps_its_script_symbol_with_no_calls():
    recs, w = run(["broken.sh", "flow.sh"])
    assert [(x["path"], x["code"], x["line"]) for x in w] == [("broken.sh", "parse-error", 2)]
    assert [s["qname"] for s in of(recs, "symbol") if s["path"] == "broken.sh"] == ["broken.sh::<script>"]
    assert not [r for r in recs if r["type"] in ("callsite", "valueSet", "data", "param")
                and r.get("symbolId", "") == sym(recs, "broken.sh::<script>")["id"]]
    assert sym(recs, "flow.sh::f")                                              # the other file is unaffected
    assert len(of(recs, "file")) == 2
    assert_valid(recs)


def test_a_missing_file_never_comes_back_as_an_empty_script(tmp_path):
    with pytest.raises(SourceRefused):
        run(["absent.sh"], root=tmp_path)


def test_nesting_beyond_the_cap_is_a_warning_not_a_crash(tmp_path):
    deep = "echo " + "$(echo " * 3000 + "x" + ")" * 3000 + "\n"
    recs, w = snippet(tmp_path, deep)
    assert [x["code"] for x in w] == ["too-deep"]
    assert [s["qname"] for s in of(recs, "symbol")] == ["s.sh::<script>"] and not of(recs, "callsite")
    shallow = "echo " + "$(echo " * 20 + "x" + ")" * 20 + "\n"
    recs, w = snippet(tmp_path, shallow, name="t.sh")
    assert w == [] and len(of(recs, "callsite")) == 21


def test_arguments_per_command_are_capped(tmp_path):
    recs, _ = snippet(tmp_path, "cmd " + " ".join(f"w{i}" for i in range(500)) + "\n")
    c = call(recs, "s.sh::<script>", "cmd")
    assert len([v for v in of(recs, "valueSet") if v["callSiteId"] == c["id"]]) == shell.MAX_ARGS
    assert_valid(recs)


def test_records_per_file_are_capped_with_a_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(shell, "MAX_RECORDS", 60)
    recs, w = snippet(tmp_path, "".join(f"echo {i}\n" for i in range(100)))
    assert [x["code"] for x in w] == ["record-cap"]
    assert 4 <= len(recs) <= 60 and len(of(recs, "callsite")) < 100
    assert_valid(recs)


def test_a_file_over_the_size_cap_fails_the_unit(tmp_path):
    (tmp_path / "big.sh").write_bytes(b"#" * (MAX_FILE_BYTES + 1))
    with pytest.raises(SourceRefused, match="over"):
        run(["big.sh"], root=tmp_path)
    (tmp_path / "ok.sh").write_bytes(b"#" * MAX_FILE_BYTES)
    assert run(["ok.sh"], root=tmp_path)[1] == []


def test_a_file_that_is_not_utf8_fails_the_unit(tmp_path):
    (tmp_path / "bad.sh").write_bytes(b"echo \xff\xfe\n")
    with pytest.raises(SourceRefused, match="UTF-8"):
        run(["bad.sh"], root=tmp_path)


@pytest.mark.parametrize("name", ["../x.sh", "/etc/passwd", "a//b.sh", "./x.sh", "a/../x.sh", "", "x\0.sh"])
def test_a_file_name_that_leaves_the_root_or_is_not_normal_is_refused(tmp_path, name):
    with pytest.raises(SourceRefused):
        run([name], root=tmp_path)


def test_a_symlink_out_of_the_root_a_directory_and_a_fifo_are_refused(tmp_path):
    outside = tmp_path / "outside.sh"
    outside.write_text("echo out\n")
    root = tmp_path / "root"
    root.mkdir()
    (root / "link.sh").symlink_to(outside)
    (root / "dir.sh").mkdir()
    os.mkfifo(root / "fifo.sh")
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(AssertionError("opening the fifo hung")))
    signal.alarm(5)                                                 # a read of a fifo with no writer would block for ever
    try:
        for name in ("link.sh", "dir.sh", "fifo.sh"):
            with pytest.raises(SourceRefused):
                run([name], root=root)
    finally:
        signal.alarm(0)
    (root / "inner.sh").write_text("echo in\n")
    (root / "ok.sh").symlink_to(root / "inner.sh")                 # a link that stays inside is read
    assert run(["ok.sh"], root=root)[1] == []
    assert stat.S_ISFIFO(os.stat(root / "fifo.sh").st_mode)


def test_a_file_listed_twice_is_a_caller_error(tmp_path):
    (tmp_path / "a.sh").write_text("echo\n")
    with pytest.raises(ValueError, match="twice"):
        run(["a.sh", "a.sh"], root=tmp_path)


def test_the_extractor_never_runs_or_follows_what_it_reads(tmp_path):
    marker = tmp_path / "ran"
    (tmp_path / "other.sh").write_text("echo other\n")
    recs, w = snippet(tmp_path, f"touch {marker}\nx=$(touch {marker})\neval 'touch {marker}'\nsource ./other.sh\n"
                                f". {tmp_path}/other.sh\n")
    assert not marker.exists() and w == []
    assert {r["path"] for r in of(recs, "file")} == {"s.sh"}               # `source` is not followed
    assert Counter(c["calleeName"] for c in of(recs, "callsite")) == \
        Counter({"touch": 3, "eval": 1, "source": 1, ".": 1})          # the third `touch` is the inner site of the eval string


_CHILD = """
import json, resource, sys
resource.setrlimit(resource.RLIMIT_AS, (900_000_000, 900_000_000))
from extractors.shell import extract_shell
recs = extract_shell(sys.argv[1], ["s.sh"], repo_id="r", unit_id="u" * 32)
print(len(recs))
"""


def _bounded(tmp_path, text: str, seconds=30) -> str:
    """Run the extractor on `text` in a child under a 900 MB address-space cap: a blow-up is an assertion, never a skip."""
    (tmp_path / "s.sh").write_text(text)
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-c", _CHILD, str(tmp_path)], cwd=CPG, capture_output=True, text=True,
                       timeout=seconds)
    assert r.returncode == 0, r.stderr[-300:]
    return r.stdout.strip()


def test_a_doubling_chain_with_a_use_after_every_step_stays_within_a_memory_bound(tmp_path):
    steps = "".join(f'  v{i}="$v{i - 1}$v{i - 1}"\n  echo "$v{i}"\n' for i in range(1, 41))
    assert int(_bounded(tmp_path, f"f() {{\n  v0=x\n{steps}}}\n")) > 80


def test_a_merged_string_past_the_literal_cap_is_a_value_set_cap_not_a_huge_const(tmp_path):
    steps = "".join(f'  v{i}="$v{i - 1}$v{i - 1}"\n' for i in range(1, 7))
    recs, _ = snippet(tmp_path, f"f() {{\n  v0=\"{'ab ' * 100}\"\n{steps}  use \"$v1\" \"$v6\"\n}}\n")
    c = call(recs, "s.sh::f", "use")
    assert values(recs, c, "a0") == [{"const": "ab " * 200}]
    assert "value-set-cap" in repr(values(recs, c, "a1"))                   # 19 200 characters: capped at assembly
    assert "ab ab ab" not in repr(values(recs, c, "a1"))


def test_hostile_self_reference_and_branching_chains_stay_within_a_memory_bound(tmp_path):
    text = ("a=" + "$a" * 5000 + "\n") + "".join(f"x{i}=$x{i - 1}$x{i - 1}\n" for i in range(1, 400)) + \
        "use \"$x399\"\n" + "echo " + "$(echo " * 30 + ")" * 30 + "\n"
    assert int(_bounded(tmp_path, text)) > 4


def test_a_uses_value_does_not_depend_on_the_uses_before_it(tmp_path):
    chain = "".join(f"  v{i}=$v{i - 1}\n" for i in range(1, 12))
    plain, _ = snippet(tmp_path, f"f() {{\n  v0=lit\n{chain}  echo \"$v11\"\n}}\n")
    interleaved = "".join(f"  v{i}=$v{i - 1}\n  echo \"$v{i}\"\n" for i in range(1, 12))
    many, _ = snippet(tmp_path, f"f() {{\n  v0=lit\n{interleaved}  echo \"$v11\"\n}}\n")
    again, _ = snippet(tmp_path, f"f() {{\n  v0=lit\n{interleaved}{interleaved.replace('v', 'w')}  echo \"$v11\"\n}}\n")
    last = lambda recs: values(recs, calls(recs, "s.sh::f", "echo")[-1], "a0")             # noqa: E731
    assert last(plain) == last(many) == last(again)


# --- determinism ----------------------------------------------------------------------------------------------

def test_two_runs_and_any_file_order_give_the_same_records():
    files = ["gquery.sh", "flow.sh", "dynamic.sh", "positional.sh", "quoting.sh"]
    a, _ = run(files)
    b, _ = run(list(reversed(files)))
    c, _ = run(files)
    key = lambda rs: [shell._canon(r) for r in rs]                                      # noqa: E731
    assert key(a) == key(b) == key(c)
    assert_valid(a)


def test_a_multi_file_unit_validates_as_one_bundle():
    recs, w = run(["gquery.sh", "flow.sh", "dynamic.sh", "positional.sh", "quoting.sh"])
    assert w == []
    assert {r["path"] for r in of(recs, "file")} == {"gquery.sh", "flow.sh", "dynamic.sh", "positional.sh", "quoting.sh"}
    assert_valid(recs)


# --- an `eval` string is shell source (P1-10 10b) ----------------------------------------------------------

S = "s.sh::<script>"
T = ("possible", "shell-dynamic", "unresolved")


def inner(recs, callee, ordinal=0):
    """(status, reason, resolution) of a call site and its argument value sets as `(a0, a1, ..)` lists."""
    c = call(recs, S, callee, ordinal)
    vs = sorted((v["arg"], v["values"]) for v in of(recs, "valueSet") if v["callSiteId"] == c["id"])
    return (c["status"], c["reason"], c["resolution"]), [v for _, v in vs]


_U = {"unknown": "runtime-built"}


def callees(recs):
    return [(c["calleeName"], c["ordinal"]) for c in sorted(of(recs, "callsite"), key=lambda r: (r["line"], r["calleeName"], r["ordinal"]))]


def test_a_double_quoted_eval_string_gives_the_inner_command_with_an_unknown_argument_beside_the_eval_site():
    recs, w = run("cpg/fixtures/system1/fx-svc/scripts/dyn.sh", root=REPO_ROOT)
    ev = call(recs, "cpg/fixtures/system1/fx-svc/scripts/dyn.sh::<script>", "eval")
    assert ev["ordinal"] == 0 and (ev["status"], ev["reason"]) == ("possible", "shell-dynamic")
    assert values(recs, ev, "a0") == [{"template": [{"const": "mcp-call "}, {"unknown": "point-derived"}]}] or \
        values(recs, ev, "a0")[0]["template"][0] == {"const": "mcp-call "}          # the eval site itself is as before
    mc = call(recs, "cpg/fixtures/system1/fx-svc/scripts/dyn.sh::<script>", "mcp-call")
    assert (mc["ordinal"], mc["status"], mc["reason"], mc["resolution"], mc["targets"]) == (0, "possible", "shell-dynamic",
                                                                                           "unresolved", [])
    assert mc["id"] == ids.callsite_id(mc["symbolId"], "mcp-call", 0) and mc["line"] == 4
    assert [(v["arg"], v["values"]) for v in of(recs, "valueSet") if v["callSiteId"] == mc["id"]] == \
        [("a0", [{"unknown": "runtime-built"}])]
    assert w == [] and flows(recs, mc) == []
    assert_valid(recs)


def test_a_single_const_eval_string_gives_a_literal_argument(tmp_path):
    recs, _ = snippet(tmp_path, "eval 'mcp-call reindex'\n")
    assert inner(recs, "mcp-call") == (T, [[{"const": "reindex"}]])
    assert len(of(recs, "callsite")) == 2 and call(recs, S, "eval")["reason"] == "shell-dynamic"
    assert_valid(recs)


def test_the_string_is_parsed_as_shell_never_split_on_spaces(tmp_path):
    recs, _ = snippet(tmp_path, "eval 'a \"b c\"'\neval \"x 'y z' w\\ v\"\n")
    assert inner(recs, "a") == (T, [[{"const": "b c"}]])
    assert inner(recs, "x") == (T, [[{"const": "y z"}], [{"const": "w v"}]])


def test_every_complete_command_of_the_string_is_a_site_and_a_terminated_last_one_takes_no_unknown(tmp_path):
    recs, _ = snippet(tmp_path, 'eval "a 1; b 2 | c && d $X; "\n')
    assert [(n, o) for n, o in callees(recs) if n != "eval"] == [("a", 0), ("b", 0), ("c", 0), ("d", 0)]
    assert inner(recs, "a")[1] == [[{"const": "1"}]] and inner(recs, "b")[1] == [[{"const": "2"}]]
    assert inner(recs, "c")[1] == []
    assert inner(recs, "d")[1] == [[{"unknown": "runtime-built"}]]


def test_a_word_counts_only_when_a_delimiter_follows_it_in_the_literal(tmp_path):
    recs, _ = snippet(tmp_path, 'eval "mcp-call$X"\neval "mcp-call re$X"\neval "mcp-call re $X"\neval "mcp-call ;$X"\n')
    assert [(n, o) for n, o in callees(recs) if n == "mcp-call"] == [("mcp-call", 0), ("mcp-call", 1), ("mcp-call", 2)]
    assert inner(recs, "mcp-call", 0)[1] == [[{"unknown": "runtime-built"}]]              # `re$X`: the open word is one unknown
    assert inner(recs, "mcp-call", 1)[1] == [[{"const": "re"}], [{"unknown": "runtime-built"}]]
    assert inner(recs, "mcp-call", 2)[1] == []                                              # `;` ended it; `$X` starts another


@pytest.mark.parametrize("src, written", [
    ('eval "$(ssh-agent)"', ["eval", "ssh-agent"]),     # starts with a non-const
    ('eval "$CMD x"', ["eval"]),
    ('eval "mcp-call$X"', ["eval"]),                    # the command word is open
    ("eval 'a \"b c'", ["eval"]),                       # an unparseable prefix: an open quote
    ("eval 'eval a b'", ["eval"]),                      # an eval inside the string
    ('eval "eval a $X"', ["eval"]),
    ("eval '$CMD x'", ["eval"]),                        # a literal `$CMD`: the name is not a plain literal
    ('eval ""', ["eval"]),
    ('eval mcp-call reindex', ["eval"]),                # several arguments: the pack reads that form itself
    ('eval "mcp-call" "$X"', ["eval"]),
    ("eval 'if t; then m x; fi'", ["eval"]),             # a compound command's body is not entered
    ("eval 'for i in a; do m $i; done'", ["eval"]),
    ("eval 'a $(b c)'", ["eval", "a"]),                 # a nested substitution is not a site
    ("eval 'x=1'", ["eval"]),                           # an assignment is not a command
    ('eval a"$X"b c', ["eval"]),
    ("CMD='mcp-call x'; eval \"$CMD\"", ["eval"]),       # a variable's value is never substituted
])
def test_no_inner_site_where_the_string_has_no_complete_literal_command(tmp_path, src, written):
    recs, _ = snippet(tmp_path, src + "\n")
    assert sorted(c["calleeName"] for c in of(recs, "callsite")) == sorted(written)


def test_the_eval_site_itself_is_unchanged_and_inner_ordinals_follow_every_written_site(tmp_path):
    recs, _ = snippet(tmp_path, 'mcp-call a\neval "mcp-call $T"\nmcp-call b\neval "mcp-call x; mcp-call y"\n')
    got = sorted((c["calleeName"], c["ordinal"], c["line"]) for c in of(recs, "callsite"))
    assert got == [("eval", 0, 2), ("eval", 1, 4), ("mcp-call", 0, 1), ("mcp-call", 1, 3),
                   ("mcp-call", 2, 2), ("mcp-call", 3, 4), ("mcp-call", 4, 4)]
    for c in of(recs, "callsite"):
        assert c["id"] == ids.callsite_id(c["symbolId"], c["calleeName"], c["ordinal"])
    assert inner(recs, "mcp-call", 0) == (("possible", "name-match", "unresolved"), [[{"const": "a"}]])
    assert inner(recs, "mcp-call", 2)[1] == [[{"unknown": "runtime-built"}]]
    assert inner(recs, "mcp-call", 4)[1] == [[{"const": "y"}]]


def test_an_inner_sites_line_is_the_line_of_its_command_inside_the_string(tmp_path):
    recs, _ = snippet(tmp_path, "echo\neval 'a 1\nb 2\n\nc'\n")
    assert [(c["calleeName"], c["line"]) for c in sorted(of(recs, "callsite"), key=lambda r: r["calleeName"])] == \
        [("a", 2), ("b", 3), ("c", 5), ("echo", 1), ("eval", 2)]


def test_a_file_without_an_eval_string_is_numbered_as_before(tmp_path):
    recs, _ = snippet(tmp_path, "mcp-call a\neval $X\nmcp-call b\nmcp-call c\n")
    assert [(c["calleeName"], c["ordinal"]) for c in of(recs, "callsite")
            if c["calleeName"] == "mcp-call"] == [("mcp-call", 0), ("mcp-call", 1), ("mcp-call", 2)]
    assert len(of(recs, "callsite")) == 4


def test_an_inner_command_naming_a_function_resolves_to_it_as_the_schema_requires(tmp_path):
    recs, _ = snippet(tmp_path, 'f() { :; }\neval "f $X"\n')
    c = call(recs, S, "f")
    assert (c["status"], c["reason"], c["resolution"], len(c["targets"])) == ("confirmed", "resolved-call", "resolved", 1)
    assert_valid(recs)


def test_an_inner_secret_and_a_secret_named_flag_are_withheld_as_in_any_command(tmp_path):
    recs, _ = snippet(tmp_path, "eval 'curl --password hunter2 --user=bob --token=abc x'\n")
    assert inner(recs, "curl")[1] == [[{"const": "--password"}], [_U], [{"const": "--user=bob"}], [_U], [{"const": "x"}]]
    assert_valid(recs)


def test_the_eval_string_is_bounded_in_characters_and_commands(tmp_path):
    many = "eval '" + "; ".join(f"c{i} x" for i in range(20)) + "'\n"
    recs, _ = snippet(tmp_path, many)
    assert [c["calleeName"] for c in of(recs, "callsite") if c["calleeName"] != "eval"] == [f"c{i}" for i in range(8)]
    pad = "a " * 2048                                              # 4,096 characters: all of it is parsed
    recs, _ = snippet(tmp_path, f"eval '{pad}; late x'\n")
    assert [c["calleeName"] for c in of(recs, "callsite")] == ["eval", "a"]            # `late` is past the cap
    recs, _ = snippet(tmp_path, "eval 'first 1; " + "#" * 5000 + "\nlate x'\n")
    assert [c["calleeName"] for c in of(recs, "callsite")] == ["eval", "first"]


def test_the_character_cap_is_exact(tmp_path):
    def sites(n):                                                  # `late` ends at character n of the string
        recs, _ = snippet(tmp_path, "eval 'first;" + " " * (n - len("first;") - len("late")) + "late'\n")
        return [c["calleeName"] for c in of(recs, "callsite")]
    assert shell.MAX_EVAL_CHARS == 4096
    assert sites(4096) == ["eval", "first", "late"]
    assert sites(4097) == ["eval", "first"]    # cut inside the word: an open command name


def test_only_the_leading_const_is_read_and_arguments_per_inner_command_are_capped(tmp_path):
    recs, _ = snippet(tmp_path, 'eval "a b $X; c d"\n')
    assert [c["calleeName"] for c in of(recs, "callsite")] == ["eval", "a"]
    recs, _ = snippet(tmp_path, "eval 'cmd " + " ".join(f"w{i}" for i in range(500)) + "'\n")
    c = call(recs, S, "cmd")
    assert len([v for v in of(recs, "valueSet") if v["callSiteId"] == c["id"]]) == shell.MAX_ARGS


def test_inner_sites_count_against_the_record_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(shell, "MAX_RECORDS", 20)
    recs, w = snippet(tmp_path, "".join(f"eval 'c{i} a b c d'\n" for i in range(10)))
    assert [x["code"] for x in w] == ["record-cap"] and len(recs) <= 20
    assert_valid(recs)


def test_an_eval_string_is_never_run_and_the_records_are_deterministic(tmp_path):
    marker = tmp_path / "ran"
    text = f'eval "touch {marker}; mcp-call $T"\n'
    a, _ = snippet(tmp_path, text)
    b, _ = snippet(tmp_path, text)
    assert not marker.exists() and a == b


def test_hostile_eval_strings_stay_within_a_memory_bound(tmp_path):
    for hostile in ("eval '" + 'a "' * 100_000 + "'\n",                       # 100k open quotes
                    'eval "' + "$(" * 100_000 + 'x"\n',                        # nested substitutions
                    "eval '" + "a" * 1_000_000 + "'\n",                       # a 1 MB const
                    "eval '" + "a; " * 300_000 + "'\n",                       # many commands
                    "eval '" + "(" * 100_000 + "'\n"):
        assert int(_bounded(tmp_path, hostile)) >= 2


# --- corpus smoke ----------------------------------------------------------------------------------------------

def _real_scripts():
    found = sorted(p.relative_to(REPO_ROOT).as_posix() for d in ("falkor-chat/scripts", "cypher-mcp")
                   for p in (REPO_ROOT / d).glob("*.sh"))
    return found


def test_the_real_scripts_parse_with_no_error_node_and_form_a_valid_bundle():
    files = _real_scripts()
    assert len(files) >= 30 and "falkor-chat/scripts/bootstrap_schema.sh" in files
    recs, w = run(files, root=REPO_ROOT)
    assert w == []                                                      # no parse-error, too-deep, record-cap
    assert {r["path"] for r in of(recs, "file")} == set(files)
    assert {s["kind"] for s in of(recs, "symbol")} == {"script", "function"}
    assert len([s for s in of(recs, "symbol") if s["kind"] == "script"]) == len(files)
    assert_valid(recs)


def test_bootstrap_schema_lifts_the_fifty_gquery_cypher_literals():
    recs, w = run(["falkor-chat/scripts/bootstrap_schema.sh"], root=REPO_ROOT)
    assert w == []
    q = "falkor-chat/scripts/bootstrap_schema.sh::gquery"
    gq = [c for c in of(recs, "callsite") if c["calleeName"] == "gquery"]
    assert len(gq) == 50 and all(c["targets"][0]["qname"] == q for c in gq)
    cypher = [values(recs, c, "a1") for c in gq]
    lead = lambda v: v[0]["const"] if "const" in v[0] else v[0]["template"][0]["const"]        # noqa: E731
    assert all(len(v) == 1 and lead(v).split()[0] in ("CREATE", "CALL") for v in cypher), cypher
    assert sum("const" in v[0] for v in cypher) >= 40                      # the rest interpolate a dimension
    assert {"const": "CREATE INDEX FOR (n:WorkflowDef) ON (n.key)"} in [v[0] for v in cypher]
    cli = call(recs, q, "redis-cli")
    assert flows(recs, cli) == [("a5", "p0", "param", "confirmed", "intra-flow"),
                                ("a6", "p1", "param", "confirmed", "intra-flow")]
    assert values(recs, cli, "a1") == [{"ref": "falkor-chat/scripts/bootstrap_schema.sh::HOST"}]
    assert_valid(recs)


def test_the_module_imports_no_bundle_rule_deciding_name():
    """The AST tripwire's own check (tests/test_writer_rules.py) runs over `extractors/**`; this pins the module is in it."""
    src = Path(shell.__file__).read_text(encoding="utf-8")
    assert "gate_argv" not in src and "bind_value_allowed" not in src and "valuehash" not in src
