"""P1-07 — the writer done-condition (F-10): `extractors/common/writer_rules.py` and `write_bundle`'s defence in depth.

Plan: `docs/plans/connected-code-graph-p1-extraction.md` § P1-07; spec: `-bundle-spec3.md` §1.6.1, §1.7.4 (W2-W4), §1.7.5.
One group per obligation, each with the spec's negative case. The launch and config writers are P1-11's; this unit
supplies the rules they call, and the guard at the bottom fails if an extractor decides them any other way.
No test reads a real key or touches the real work area.
"""
from __future__ import annotations

import ast
import os
import stat
from pathlib import Path

import pytest

from ccg import bundle, ids
from ccg.valuehash import entry_hmac
from extractors.common import writer_rules as wr

CPG = Path(__file__).resolve().parent.parent
KEY = bytes(range(64, 96))
DECL = frozenset({"falkordb", "server", "api"})


def bind(key_path="x.url", value="http://localhost:8000/", **kw):
    kw.setdefault("binding_role", True)
    kw.setdefault("declared_services", DECL)
    return wr.bind_value(key_path, value, **kw)


def default(env_var="FALKORDB_HOST", value="falkordb", **kw):
    kw.setdefault("binding_role", True)
    kw.setdefault("declared_services", DECL)
    return wr.launch_default(env_var, value, **kw)


# --- W2: single-label hosts are declared services -----------------------------------------

@pytest.mark.parametrize("value", [
    "falkordb:6379", "redis://falkordb:6379", "http://api:8000", "http://server:8200/mcp", "falkordb",
    "localhost:8000", "http://localhost:8000/mcp", "8200", "ntfy.sh", "10.0.0.5:8080",       # a `url` to an IP is W6's (spec5 §6.2)
])
def test_w2_declared_or_non_single_label_hosts_bind(value):
    assert bind("x.addr", value) == value


@pytest.mark.parametrize("value", [
    "acme", "acme:6379", "redis://acme:6379", "http://worker:8000", "http://worker:8000/mcp",
])
def test_w2_undeclared_single_label_hosts_never_bind(value):
    assert bind("x.addr", value) is None


def test_w2_declared_set_is_per_call_and_case_insensitive():
    assert bind("x.addr", "worker:8000", declared_services=frozenset({"Worker"})) == "worker:8000"
    assert bind("x.addr", "worker:8000", declared_services=frozenset()) is None


def test_w2_a_service_declared_elsewhere_does_not_count():
    assert bind("x.addr", "falkordb:6379", declared_services=frozenset({"falkordb2"})) is None


def test_w2_governs_launch_fact_default():
    assert default("FALKORCHAT_WS_ID", "acme", declared_services=frozenset()) is None
    assert default("FALKORCHAT_WS_ID", "acme", declared_services=frozenset({"acme"})) == "acme"
    assert default("AGENT_TEAM_PORT", "8200", declared_services=frozenset()) == "8200"     # spec's positive example
    assert default("ROOT_PHRASE", "hunter") is None                                        # spec's negative example


def test_binding_role_is_required_for_bind_value_and_default():
    assert bind(binding_role=False) is None
    assert default(binding_role=False) is None
    assert bind(binding_role=True) == "http://localhost:8000/"


def test_default_must_pass_the_value_grammar():
    assert default("X_PORT", "a b") is None
    assert default("X_PORT", "") is None
    assert default("X_PORT", "x" * 600) is None
    assert default("X_PORT", "99999") is None                  # not a port, not a host


def test_default_env_var_must_pass_the_name_rule():
    assert default("ADMIN_PASSWORD", "8200") is None
    assert default("API_TOKEN", "8200") is None
    assert default("ADMIN_PORT", "8200") == "8200"


def test_bind_value_keeps_w1_name_rule_and_grammar():
    assert bind("services.app.command[2]", "http://localhost:8000/") is None      # W1
    assert bind("x.password", "8200") is None                                      # §1.7.3
    assert bind("x.url", "not a value") is None                                    # §1.7.2


# --- W3: no interpolation; aliases judged at both ends -------------------------------------

@pytest.mark.parametrize("value", [
    "${PORT}", "$PORT", "localhost:${PORT}", "http://localhost:${PORT}/mcp", "${HOST:-falkordb}", "http://$HOST:8000",
])
def test_w3_interpolation_never_binds_nor_defaults(value):
    assert bind("x.url", value) is None
    assert default("X_PORT", value) is None


def test_w3_alias_binds_only_if_both_ends_pass_the_name_rule():
    assert bind("x.url", "http://localhost:8000/", via_alias=True, anchor_key_path="base.url") == "http://localhost:8000/"


def test_w3_alias_with_a_failing_anchor_does_not_bind():
    assert bind("x.url", "http://localhost:8000/", via_alias=True, anchor_key_path="base.secret_url") is None


def test_w3_alias_with_a_failing_alias_key_path_does_not_bind():
    assert bind("x.password", "8200", via_alias=True, anchor_key_path="base.port") is None


def test_w3_alias_without_a_known_anchor_does_not_bind():
    assert bind("x.url", "http://localhost:8000/", via_alias=True) is None


def test_w3_a_plain_value_ignores_anchor_argument():
    assert bind("x.url", "http://localhost:8000/", anchor_key_path="base.secret_url") == "http://localhost:8000/"


# --- W4: withholding is always allowed ------------------------------------------------------

def test_w4_a_writer_may_withhold_a_value_that_would_bind():
    assert bind() == "http://localhost:8000/"
    assert bind(withhold=True) is None
    assert default(withhold=True) is None


def test_w4_withheld_fields_are_absent_not_null():
    assert wr.bind_fields("x.url", "http://localhost:8000/", binding_role=True, declared_services=DECL) == {
        "bindValue": "http://localhost:8000/"}
    assert wr.bind_fields("x.url", "http://localhost:8000/", binding_role=True, declared_services=DECL,
                          withhold=True) == {}
    assert wr.bind_fields("x.password", "8200", binding_role=True, declared_services=DECL) == {}


def test_w4_default_hash_is_present_even_when_default_is_withheld():
    lid = ids.launch_fact_id("r", "Dockerfile", "env-default", 0)
    kept = wr.default_fields(KEY, lid, "AGENT_TEAM_PORT", "8200", binding_role=True, declared_services=DECL)
    held = wr.default_fields(KEY, lid, "AGENT_TEAM_PORT", "8200", binding_role=True, declared_services=DECL,
                             withhold=True)
    refused = wr.default_fields(KEY, lid, "ROOT_PHRASE", "hunter", binding_role=True, declared_services=DECL)
    assert kept["default"] == "8200" and "default" not in held and "default" not in refused
    assert kept["defaultHash"] == held["defaultHash"] == wr.default_hash(KEY, lid, "8200")
    assert refused == {"defaultHash": wr.default_hash(KEY, lid, "hunter")}
    assert "hunter" not in repr(refused)


# --- SA-1 writer side: argv only through gate_argv ------------------------------------------

def test_launch_argv_fields_are_gate_argv_output():
    argv = ["redis-cli", "AUTH", "hunter2", "--port", "6379"]
    got = wr.launch_argv_fields(argv)
    gated = bundle.gate_argv(argv)
    assert gated != argv                                       # the gate really changed something
    assert [got["command"], *got["args"]] == gated
    assert "hunter2" not in repr(got)


def test_launch_argv_fields_keep_order_and_fail_command_closed():
    got = wr.launch_argv_fields(["bad command", "--b", "2", "--a", "1"])
    assert got == {"command": bundle.REDACTED, "args": ["--b", "2", "--a", "1"]}


def test_launch_argv_fields_empty_args_and_empty_argv():
    assert wr.launch_argv_fields(["uvicorn"]) == {"command": "uvicorn", "args": []}
    with pytest.raises(ValueError, match="argv"):
        wr.launch_argv_fields([])


def test_launch_argv_fields_forces_the_next_element_after_a_secret_flag():
    got = wr.launch_argv_fields(["app", "--token", "abc123"])
    assert got["args"] == ["--token", bundle.REDACTED]


# --- the guard: a tripwire, plus a per-obligation backstop ----------------------------------
#
# Static: no module under extractors/** other than common/writer_rules.py writes a field this unit owns or reaches the
# rule-deciding names. It is a TRIPWIRE: it reads literal and constant-foldable forms. It cannot see `dict(zip(..))`,
# `r.update(var)`, a `**splat` of a built mapping, a key computed from data, `dict.fromkeys`, or a dataclass field.
# Backstop for those, per obligation: `writer_rules.audit_records` (below), which a writer's own test runs over the
# records it produced (grammar, name rule, W1 and the argv fixed point are also re-checked by `bundle.validate`).

_OWNED_FIELDS = frozenset({"command", "args", "bindValue", "default", "valueHash", "defaultHash", "bindParts"})
_EXEMPT = frozenset({("decorator", "args")})       # a decorator record's own `args`: the one legitimate overlap
_RULE_NAMES = frozenset({"bind_value_allowed", "value_grammar_ok", "gate_argv"})
_KEY_NAMES = frozenset({"hmac_key", "key_path", "bind_key_check", "KEY_ROOT", "credentials_dir", "entry_hmac"})
_FORBIDDEN_MODULES = ("ccg.valuehash",)


def _fold(node: ast.AST, names: dict[str, str]) -> str | None:
    """The constant string a key expression folds to (literal, `+`, f-string of constants, a constant-bound name)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        l, r = _fold(node.left, names), _fold(node.right, names)
        return None if l is None or r is None else l + r
    if isinstance(node, ast.JoinedStr):
        parts = [_fold(v.value if isinstance(v, ast.FormattedValue) else v, names) for v in node.values]
        return None if any(p is None for p in parts) else "".join(parts)
    return None


def owned_field_writes(source: str) -> list[int]:
    """Line numbers where `source` writes a field of `_OWNED_FIELDS`: a dict key, a keyword, a subscript store, a
    `dict(...)`/`.update(...)` keyword, `setdefault`/`__setitem__`/`setattr` naming one, a `(name, value)` pair inside `dict([...])`. Keys are folded
    (concatenation, f-string of constants, a name bound once to a constant string). Reads are not writes. A dict
    literal is exempt only for a (constant type, field) pair in `_EXEMPT`."""
    tree = ast.parse(source)
    names = {t.id: n.value.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
             and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)
             for t in n.targets if isinstance(t, ast.Name)}
    owned = lambda node: _fold(node, names) in _OWNED_FIELDS      # noqa: E731
    hits: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            typ = next((v.value for k, v in zip(node.keys, node.values)
                        if isinstance(k, ast.Constant) and k.value == "type" and isinstance(v, ast.Constant)), None)
            hits += [k.lineno for k in node.keys
                     if k is not None and owned(k) and (typ, _fold(k, names)) not in _EXEMPT]
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del)) and owned(node.slice):
            hits.append(node.lineno)
        elif isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Name) and f.id == "dict") or (isinstance(f, ast.Attribute) and f.attr == "update"):
                hits += [k.value.lineno for k in node.keywords if k.arg in _OWNED_FIELDS]   # not `max(default=0)`
            if isinstance(f, ast.Attribute) and f.attr in ("setdefault", "__setitem__") and node.args \
                    and owned(node.args[0]):
                hits.append(node.lineno)
            elif isinstance(f, ast.Name) and f.id == "setattr" and len(node.args) >= 2 and owned(node.args[1]):
                hits.append(node.lineno)
            elif isinstance(f, ast.Name) and f.id == "dict":
                hits += [e.lineno for a in node.args for e in ast.walk(a)
                         if isinstance(e, ast.Tuple) and e.elts and owned(e.elts[0])]
    return hits


def _private_function(name: str) -> bool:
    """A private bundle name that carries logic (`_port`, `_bind_grammar`); an UPPER_CASE private constant
    (`_UUID`, a regex the masker reads) is data, not a rule decision."""
    return name.startswith("_") and not name.startswith("__") and not name[1:].isupper()


def forbidden_references(source: str) -> list[str]:
    """What `source` reaches that only writer_rules may: the rule-deciding bundle names, bundle's private names, the
    key accessors of ccg.atrest, ccg.valuehash. Importing `ccg.atrest`/`ccg.bundle` for their write, validate and
    screen functions is allowed (the normalizer does)."""
    out: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod in _FORBIDDEN_MODULES:
                out.append(mod)
            for a in node.names:
                if mod == "ccg" and a.name == "valuehash":
                    out.append("ccg.valuehash")
                if mod == "ccg.bundle" and (a.name in _RULE_NAMES or _private_function(a.name) or a.name == "*"):
                    out.append(f"ccg.bundle.{a.name}")
                if mod == "ccg.atrest" and (a.name in _KEY_NAMES or a.name == "*"):
                    out.append(f"ccg.atrest.{a.name}")
        elif isinstance(node, ast.Import):
            out += [a.name for a in node.names if a.name in _FORBIDDEN_MODULES]
        elif isinstance(node, ast.Attribute):
            if node.attr in _RULE_NAMES or node.attr in _KEY_NAMES or (
                    _private_function(node.attr) and isinstance(node.value, ast.Name) and node.value.id == "bundle"):
                out.append(node.attr)
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str) and node.args[0].value in _FORBIDDEN_MODULES:
            out.append(node.args[0].value)          # importlib.import_module("ccg.valuehash"), __import__(...)
    return out


# Planted-offender rows (root REVIEW.md): every form the tripwire claims to catch, per owned field.
_FORMS = [
    '{{"type": "t", "{f}": v}}', '{{"{f}": v}}', "dict({f}=v)", 'r["{f}"] = v', 'r["{f}"] += v',
    'r.update({{"{f}": v}})', 'r.update({f}=v)', 'r.setdefault("{f}", v)', 'r.__setitem__("{f}", v)',
    '{{**base, "{f}": v}}', 'setattr(r, "{f}", v)', 'dict([("{f}", v)])', 'dict((("{f}", v),))',
    'r["{a}" + "{b}"] = v', '{{"{a}" "{b}": v}}', 'r[f"{a}{b}"] = v', 'K = "{f}"\nr[K] = v', 'K = "{f}"\n{{K: v}}',
    '{{"type": "launchFact", "{f}": v}}', '{{"type": "configEntry", "{f}": v}}', '{{"type": kind, "{f}": v}}',
    "def f():\n    return {{'{f}': v}}",
]


_SEVEN = ["command", "args", "bindValue", "default", "valueHash", "defaultHash", "bindParts"]     # literal: the six of P1-07 plus P1-14's


def test_the_guards_owned_field_set_is_exactly_the_seven_of_the_plan():
    assert sorted(_OWNED_FIELDS) == sorted(_SEVEN)       # a shrunk set must not just shrink the parametrisation below


@pytest.mark.parametrize("field", _SEVEN)
@pytest.mark.parametrize("form", _FORMS)
def test_guard_flags_every_form_of_writing_an_owned_field(field, form):
    src = form.format(f=field, a=field[:2], b=field[2:])
    assert owned_field_writes(src), src


def test_guard_flags_bind_parts_in_a_typed_record_and_does_not_flag_a_read():
    assert owned_field_writes('r = {"type": "configEntry", "bindParts": p}')
    assert owned_field_writes('r["bind" "Parts"] = p')
    assert not owned_field_writes('x = r["bindParts"]\ny = r.get("bindParts")')


def test_writer_rules_writes_bind_parts_only_inside_bind_parts_fields_which_asks_bind_parts_allowed():
    src = (CPG / "extractors" / "common" / "writer_rules.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "bind_parts_fields")
    writes = [ln for ln in owned_field_writes(src) if any(
        isinstance(n, ast.Dict) and n.lineno <= ln <= n.end_lineno
        and any(isinstance(k, ast.Constant) and k.value == "bindParts" for k in n.keys) for n in ast.walk(tree))]
    assert writes and all(fn.lineno <= ln <= fn.end_lineno for ln in writes)
    assert any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "bind_parts_allowed"
               for c in ast.walk(fn))


def test_guard_exempts_only_the_decorator_args_pair():
    assert not owned_field_writes('r = {"type": "decorator", "args": a}')
    assert owned_field_writes('r = {"type": "decorator", "command": a}')           # a typed dict is not a free pass
    assert owned_field_writes('r = {"type": "decorator", "bindValue": a}')
    assert owned_field_writes('r = {"type": "launchFact", "args": a}')


@pytest.mark.parametrize("src", [
    'x = rec["args"]', 'x = rec.get("command")', 'r = {"cmd": c, "argv": a}', 'r = {"kind": "command"}',
    'r["port"] = 1', 'r = {"type": "decorator", "args": a}', 'x = rec["bindValue"]', 'f(default)', 'K = 3\nr[K] = 1',
    'max(xs, default=0)', 'ap.add_argument("--a", default=1)', 'f(command=c, args=a)',
])
def test_guard_does_not_flag_reads_or_other_names(src):
    assert not owned_field_writes(src)


@pytest.mark.parametrize("src", [
    'dict(zip(names, vals))', 'r.update(extra)', '{"type": "launchFact", **build()}', 'r[name] = v',
    'dict.fromkeys(names)', 'r[k.lower()] = v',
])
def test_guard_is_documented_blind_to_computed_keys(src):
    """The tripwire's stated limit: these are left to `audit_records`. If one starts being caught, shrink the list."""
    assert not owned_field_writes(src)


@pytest.mark.parametrize("src", [
    "from ccg.valuehash import entry_hmac", "from ccg import valuehash", "import ccg.valuehash",
    "from ccg.bundle import bind_value_allowed", "from ccg.bundle import value_grammar_ok",
    "from ccg.bundle import gate_argv", "from ccg.bundle import _bind_grammar", "from ccg.bundle import _port as p",
    "from ccg.bundle import *", "from ccg.atrest import hmac_key", "from ccg.atrest import key_path",
    "from ccg.atrest import *", "from ccg import bundle\nbundle.gate_argv(a)", "from ccg import bundle\nbundle._port(x)",
    "from ccg import bundle\nbundle.bind_value_allowed(k, v)", "from ccg import atrest\natrest.hmac_key('s')",
    "from ccg import atrest\natrest.KEY_ROOT", 'import importlib\nimportlib.import_module("ccg.valuehash")',
    '__import__("ccg.valuehash")',
])
def test_guard_flags_every_form_of_reaching_the_rule_deciding_names(src):
    assert forbidden_references(src), src


@pytest.mark.parametrize("src", [
    "from ccg import bundle\nbundle.screen_fails(x) or bundle.name_fails(x)", "from ccg.bundle import JS_TEST_EXTENSIONS",
    "from ccg import atrest\natrest.ensure_private(p, a)\natrest.write_bundle(s, u, b, r)", "from ccg import ids, atrest",
    "from ccg.bundle import validate, dumps", "from ccg import bundle\nbundle._UUID.match(x) and bundle._HEXCH",
    "from ccg.bundle import _UUID",
])
def test_guard_allows_what_the_normalizer_legitimately_uses(src):
    assert not forbidden_references(src)


def _extractor_sources() -> list[Path]:
    return sorted(p for p in (CPG / "extractors").rglob("*.py")
                  if "__pycache__" not in p.parts and not (p.name == "writer_rules.py" and p.parent.name == "common"))


def test_the_scan_covers_the_real_extractors():
    names = {p.name for p in _extractor_sources()}
    assert {"assemble_bundle.py", "run.py"} <= names and "writer_rules.py" not in names


def test_only_writer_rules_writes_owned_fields_in_the_extractors():
    offenders = {str(p.relative_to(CPG)): ln for p in _extractor_sources()
                 if (ln := owned_field_writes(p.read_text(encoding="utf-8")))}
    assert not offenders, f"owned fields must come from writer_rules (SA-1, W2, W3, §1.7.5): {offenders}"


def test_extractors_decide_the_rules_only_through_writer_rules():
    offenders = {str(p.relative_to(CPG)): r for p in _extractor_sources()
                 if (r := forbidden_references(p.read_text(encoding="utf-8")))}
    assert not offenders, f"these names are reached only through extractors.common.writer_rules: {offenders}"


def test_writer_rules_itself_gates_every_argv_field_it_writes():
    src = (CPG / "extractors" / "common" / "writer_rules.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "launch_argv_fields")
    assert any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "gate_argv"
               for c in ast.walk(fn))
    argv_writes = [ln for ln in owned_field_writes(src) if any(
        isinstance(n, ast.Dict) and n.lineno <= ln <= n.end_lineno
        and any(isinstance(k, ast.Constant) and k.value in ("command", "args") for k in n.keys)
        for n in ast.walk(tree))]
    assert argv_writes and all(fn.lineno <= ln <= fn.end_lineno for ln in argv_writes)


# --- the behavioural backstop: audit_records over what a writer produced ----------------------

A_REPO, A_PATH = "r1", "falkor-chat/compose.yaml"


def _ce(key_path="x.port", value="8200", text=None, **kw):
    rid = ids.config_entry_id(A_REPO, A_PATH, key_path)
    return {"type": "configEntry", "id": rid, "keyPath": key_path, "bindValue": value,
            "valueHash": wr.value_hash(KEY, rid, value if text is None else text), **kw}


def _lf_default(n=0, env="AGENT_TEAM_PORT", default="8200", **kw):
    lid = ids.launch_fact_id(A_REPO, A_PATH, "env-default", n)
    return {"type": "launchFact", "id": lid, "kind": "env-default", "envVar": env, "default": default,
            "defaultHash": wr.default_hash(KEY, lid, default), **kw}


def _lf_cmd(argv=("uvicorn", "app:app", "--port", "8000")):
    return {"type": "launchFact", "id": ids.launch_fact_id(A_REPO, A_PATH, "command", 0), "kind": "command",
            **wr.launch_argv_fields(list(argv))}


def audit(records, **kw):
    kw.setdefault("declared_services", DECL)
    return wr.audit_records(records, key=KEY, **kw)


def test_audit_accepts_what_the_writer_rules_produce():
    lid = ids.launch_fact_id(A_REPO, A_PATH, "env-default", 5)
    held = {"type": "launchFact", "id": lid, "kind": "env-default", "envVar": "ROOT_PHRASE",
            **wr.default_fields(KEY, lid, "ROOT_PHRASE", "hunter", binding_role=True, declared_services=DECL)}
    recs = [_ce(), _ce("svc.host", "falkordb"), _lf_default(), _lf_cmd(), held, {"type": "file", "id": "0" * 32}]
    assert "default" not in held
    assert audit(recs) == []


def test_audit_flags_a_hand_built_undeclared_single_label_bind_value():              # W2, bypassing bind_fields
    assert audit([_ce("svc.host", "acme")])
    assert audit([_ce("x.url", "http://worker:8000")])


def test_audit_flags_a_hand_built_undeclared_default():                              # W2, the spec's own example
    assert audit([_lf_default(env="ROOT_PHRASE", default="hunter")])


def test_audit_flags_grammar_name_rule_and_interpolation():                          # W3
    assert audit([_ce("x.port", "${PORT}")])
    assert audit([_ce("x.password", "8200")])
    assert audit([_ce("services.app.command[2]", "8200")])                          # W1
    assert audit([_lf_default(env="ADMIN_PASSWORD")])


def test_audit_flags_a_value_hash_under_the_wrong_domain():
    r = _ce()
    r["valueHash"] = entry_hmac(KEY, "x.port", "8200")                            # keyPath domain
    assert audit([r])
    other = ids.config_entry_id(A_REPO, A_PATH, "other.port")
    r["valueHash"] = wr.value_hash(KEY, other, "8200")                               # another entry's id
    assert audit([r])


def test_audit_flags_a_default_hash_under_the_wrong_domain():
    r = _lf_default()
    r["defaultHash"] = wr.default_hash(KEY, ids.launch_fact_id(A_REPO, A_PATH, "env-default", 9), "8200")
    assert audit([r])
    r["defaultHash"] = wr.value_hash(KEY, ids.config_entry_id(A_REPO, A_PATH, "k"), "8200")
    assert audit([r])


def test_audit_checks_value_hash_of_withheld_values_against_the_extractors_texts():
    r = _ce(text="8200")
    del r["bindValue"]
    assert audit([r], texts={r["id"]: "8200"}) == []
    assert audit([r], texts={r["id"]: "8201"})                                       # hashed a different text


def test_audit_requires_default_hash_whenever_an_env_default_exists():
    r = _lf_default()
    del r["defaultHash"]
    assert audit([r])
    held = _lf_default()
    del held["default"]
    assert audit([held]) == []                                                       # withheld default keeps its hash
    del held["defaultHash"]
    assert audit([held])


def test_audit_flags_a_missing_or_malformed_value_hash_without_any_text_to_recompute():
    r = _ce()
    del r["bindValue"]
    r["valueHash"] = "ABC"
    assert audit([r])
    del r["valueHash"]
    assert audit([r])
    r["valueHash"] = wr.value_hash(KEY, r["id"], "8200")
    assert audit([r]) == []


def test_audit_flags_default_fields_on_a_launch_fact_that_is_not_env_default():
    port = {"type": "launchFact", "id": ids.launch_fact_id(A_REPO, A_PATH, "port", 0), "kind": "port", "port": 80}
    assert audit([port]) == []
    assert audit([{**port, "defaultHash": "0" * 64}]) and audit([{**port, "default": "80"}])


def test_audit_flags_a_record_id_that_is_not_32_hex():
    for bad in ("x", "G" * 32, "A" * 32):          # the hash is consistent with the bad id: only the shape check can flag it
        ce = {**_ce(), "id": bad}
        ce["valueHash"] = entry_hmac(KEY, bad, "8200")
        assert audit([ce]), bad
        lf = {**_lf_default(), "id": bad}
        lf["defaultHash"] = entry_hmac(KEY, bad, "8200")
        assert audit([lf]), bad


def test_audit_flags_null_instead_of_absent_and_malformed_hashes():                 # W4
    assert audit([_ce(bindValue=None)])
    nulled = _lf_default()
    nulled["default"] = None
    assert audit([nulled])
    r = _ce()
    r["valueHash"] = "ABC"
    assert audit([r])


def test_audit_flags_an_argv_that_is_not_a_gate_fixed_point():                      # SA-1
    r = _lf_cmd()
    r["args"] = ["redis-cli", "AUTH", "hunter2"]
    assert audit([r])
    r = _lf_cmd()
    r["command"] = "bad command"
    assert audit([r])


def test_audit_messages_never_carry_record_text():
    msgs = audit([_ce("svc.host", "acme-secret-xyz"), _lf_default(env="ROOT_PHRASE", default="hunter-xyz")])
    assert msgs and not any("secret-xyz" in m or "hunter-xyz" in m for m in msgs)


# --- fail-closed on hostile input (W4) ---------------------------------------------------------

@pytest.mark.parametrize("key_path", ["a[", "[01]", '["\\a"]', '["\\u00"]', "a..b[", '["unterminated', "[-1]"])
def test_bind_value_withholds_on_a_key_path_that_does_not_parse(key_path):
    assert bind(key_path, "8200") is None
    assert wr.bind_fields(key_path, "8200", binding_role=True, declared_services=DECL) == {}
    assert bind(key_path, "8200", via_alias=True, anchor_key_path="ok.port") is None
    assert bind("ok.port", "8200", via_alias=True, anchor_key_path=key_path) is None


@pytest.mark.parametrize("key_path", ['["\\a"]', '["\\u00"]'])
def test_keypath_parse_errors_are_bundle_errors(key_path):
    with pytest.raises(bundle.BundleError):
        bundle._keypath_segments(key_path)


def test_a_huge_bad_key_path_gives_a_capped_message():
    for bad in ("a[" + "x" * 300_000, "[01]" * 75_000, '["\\a' + "b" * 300_000 + '"]'):
        with pytest.raises(bundle.BundleError) as e:
            bundle._keypath_segments(bad)
        assert len(str(e.value)) <= 200
        assert bind(bad, "8200") is None


def test_error_messages_escape_and_cap_repo_text():
    with pytest.raises(bundle.BundleError) as e:
        bundle._keypath_segments("a[\n\x00‮" + "z" * 500)
    msg = str(e.value)
    assert "\n" not in msg and "\x00" not in msg and "‮" not in msg and len(msg) <= 200
    with pytest.raises(bundle.BundleError) as e3:
        bundle._keypath_segments("a[caf\u00e9\u4e2d")
    assert str(e3.value).replace("§", "").isascii()           # the spec sign is the message's own
    with pytest.raises(ValueError) as e2:
        wr.value_hash(KEY, "k\n" + "z" * 300_000, "v")
    assert len(str(e2.value)) <= 200 and "\n" not in str(e2.value)


def test_a_lone_surrogate_is_withheld_not_raised_by_the_field_helpers():
    lid = ids.launch_fact_id("r", "Dockerfile", "env-default", 0)
    assert wr.bind_fields("x.port", "\ud800", binding_role=True, declared_services=DECL) == {}
    assert wr.default_fields(KEY, lid, "X_PORT", "\ud800", binding_role=True, declared_services=DECL) == {}


def test_entry_hmac_raises_a_typed_capped_error_on_a_lone_surrogate():
    from ccg.valuehash import UnhashableValue
    with pytest.raises(UnhashableValue) as e:
        wr.value_hash(KEY, "0" * 32, "ZQ\ud800XY")
    assert issubclass(UnhashableValue, ValueError) and len(str(e.value)) <= 80
    assert "ZQ" not in str(e.value) and "XY" not in str(e.value) and "ud800" not in str(e.value)


@pytest.mark.parametrize("text,ok", [("", True), ("8200", True), ("caf\u00e9", True), ("\U0001f600", True),
                                     ("\ud800", False), ("ab\udfffcd", False), (b"x", False), (None, False)])
def test_encodable_is_the_one_utf8_test(text, ok):
    assert wr.encodable(text) is ok


def test_unencodable_default_and_value_are_withheld_by_every_helper():
    lid = ids.launch_fact_id("r", "Dockerfile", "env-default", 0)
    for bad in ("\ud800", "8200\udc00"):
        assert bind("x.port", bad) is None and default("X_PORT", bad) is None
        assert wr.default_fields(KEY, lid, "X_PORT", bad, binding_role=True, declared_services=DECL) == {}
    assert bind("x\ud800.port", "8200") is None and default("X\ud800", "8200") is None
    assert bind("x.port", "8200", via_alias=True, anchor_key_path="a\ud800.b") is None


def test_single_label_host_is_private():
    assert not hasattr(wr, "single_label_host") and "single_label_host" not in wr.__all__
    assert wr._single_label_host("http://worker:8000") == "worker"


# --- write_bundle: defence in depth ----------------------------------------------------------

def _records():
    from test_bundle import rich_bundle      # noqa: PLC0415  (the shared fixture of the validator's tests)
    return rich_bundle()


@pytest.fixture
def umask():
    old = os.umask(0)
    yield
    os.umask(old)


def _mode(p: Path) -> int:
    return stat.S_IMODE(p.stat().st_mode)


def test_write_bundle_is_0600_under_umask_zero(tmp_path, umask):
    out = tmp_path / "a.bundle.jsonl"
    bundle.write_bundle(out, _records())
    assert _mode(out) == 0o600
    assert bundle.read_bundle(out)


def test_write_bundle_is_0600_under_a_restrictive_umask(tmp_path):
    old = os.umask(0o277)               # open(2)'s 0600 alone would give 0400: fchmod makes it exactly 0600
    try:
        out = tmp_path / "a.bundle.jsonl"
        bundle.write_bundle(out, _records())
    finally:
        os.umask(old)
    assert _mode(out) == 0o600


def test_write_bundle_replaces_a_wider_existing_file_with_0600(tmp_path, umask):
    out = tmp_path / "a.bundle.jsonl"
    out.write_bytes(b"old")
    out.chmod(0o666)
    bundle.write_bundle(out, _records())
    assert _mode(out) == 0o600


def test_write_bundle_never_writes_through_a_planted_temp_name(tmp_path, monkeypatch, umask):
    victim = tmp_path / "victim.txt"
    victim.write_text("keep")
    monkeypatch.setattr(bundle.secrets, "token_hex", lambda n=8: "feedface")
    out = tmp_path / "a.bundle.jsonl"
    (tmp_path / f".{out.name}.feedface.tmp").symlink_to(victim)        # O_EXCL refuses an existing name
    with pytest.raises(FileExistsError):
        bundle.write_bundle(out, _records())
    assert victim.read_text() == "keep" and not out.exists()


def test_write_bundle_leaves_no_temp_file_when_validation_fails(tmp_path):
    out = tmp_path / "a.bundle.jsonl"
    with pytest.raises(bundle.BundleError):
        bundle.write_bundle(out, [{"type": "secretSighting"}])
    assert list(tmp_path.iterdir()) == []


def test_write_bundle_leaves_no_temp_file_when_the_rename_fails(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("replace failed")
    monkeypatch.setattr(bundle.os, "replace", boom)
    with pytest.raises(OSError, match="replace failed"):
        bundle.write_bundle(tmp_path / "a.bundle.jsonl", _records())
    assert list(tmp_path.iterdir()) == []


def test_write_bundle_opens_the_temp_file_exclusively_with_mode_0600(tmp_path, monkeypatch):
    """The creation mode itself is 0600 (no window wider than that before fchmod); O_NOFOLLOW is redundant with O_EXCL
    (an equivalent mutant, so it is not asserted)."""
    seen = []
    real = os.open

    def spy(path, flags, mode=0o777, *a, **k):
        seen.append((flags, mode))
        return real(path, flags, mode, *a, **k)
    monkeypatch.setattr(bundle.os, "open", spy)
    bundle.write_bundle(tmp_path / "a.bundle.jsonl", _records())
    (flags, mode), = seen
    assert mode == 0o600
    assert flags & os.O_EXCL and flags & os.O_CREAT and flags & os.O_WRONLY and not flags & os.O_TRUNC


def test_write_bundle_closes_the_descriptor_and_removes_the_temp_when_fdopen_fails(tmp_path, monkeypatch):
    def fds():
        return len(os.listdir("/proc/self/fd"))
    before = fds()
    monkeypatch.setattr(bundle.os, "fdopen", lambda *a, **k: (_ for _ in ()).throw(OSError("fdopen failed")))
    with pytest.raises(OSError, match="fdopen failed"):
        bundle.write_bundle(tmp_path / "a.bundle.jsonl", _records())
    assert list(tmp_path.iterdir()) == [] and fds() == before


def test_write_bundle_never_unlinks_a_file_it_did_not_create(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle.secrets, "token_hex", lambda n=8: "feedface")
    planted = tmp_path / ".a.bundle.jsonl.feedface.tmp"
    planted.write_text("not ours")
    with pytest.raises(FileExistsError):
        bundle.write_bundle(tmp_path / "a.bundle.jsonl", _records())
    assert planted.read_text() == "not ours"


# --- B1: an args-only argv has no program, so argv[0] is redacted and every element is an argument ------------

NO_PROGRAM = [
    (["--password", "hunter2"], "hunter2"),
    (["--password", "hunter2lower", "--port", "8000"], "hunter2lower"),
    (["-p", "hunter2lower"], "hunter2lower"),
    (["--api-key", "HUNTER2UPPER"], "HUNTER2UPPER"),
    (["--api-key", "Abc"], "Abc"),
    (["--token", "x7"], "x7"),
    (["-pHunter2lower"], "Hunter2lower"),
]


@pytest.mark.parametrize("argv,secret", NO_PROGRAM)
def test_flag_first_argv_never_leaks_and_command_is_the_redaction_sentinel(argv, secret):
    got = wr.launch_argv_fields(argv)
    assert got["command"] == bundle.REDACTED
    assert secret not in repr(got)
    assert len(got["args"]) == len(argv)                     # every element is kept as an argument, in order


@pytest.mark.parametrize("argv,secret", NO_PROGRAM)
def test_program_known_false_gives_the_same_result_as_a_leading_flag(argv, secret):
    assert wr.launch_argv_fields(argv, program_known=False) == wr.launch_argv_fields(argv)


@pytest.mark.parametrize("argv", [["8000"], ["hunter2lower"], ["serve", "--password", "hunter2lower"], ["x"]])
def test_a_dockerfile_cmd_beside_an_entrypoint_has_no_program(argv):         # explicit: no leading dash to detect
    got = wr.launch_argv_fields(argv, program_known=False)
    assert got["command"] == bundle.REDACTED and len(got["args"]) == len(argv)
    if "--password" in argv:                         # a bare unflagged positional is §1.6.1's named residual
        assert "hunter2lower" not in repr(got)


def test_flag_first_list_with_ordinary_flags_stays_intact():
    argv = ["--port", "8000", "--host", "x"]
    assert wr.launch_argv_fields(argv) == {"command": bundle.REDACTED, "args": argv}


def test_a_known_program_is_unchanged():
    argv = ["uvicorn", "app:app", "--port", "8000"]
    assert wr.launch_argv_fields(argv) == {"command": "uvicorn", "args": argv[1:]}
    assert wr.launch_argv_fields(argv, program_known=True) == wr.launch_argv_fields(argv)
    got = wr.launch_argv_fields(["redis-cli", "--password", "hunter2lower"])
    assert got["command"] == "redis-cli" and "hunter2lower" not in repr(got)


@pytest.mark.parametrize("argv", [["-jar", "app.jar"], ["--"], ["-"], ["-<redacted>", "x"], ["--password=x"]])
def test_command_never_holds_a_flag(argv):
    assert not wr.launch_argv_fields(argv)["command"].startswith("-")


def test_a_flag_first_result_is_a_gate_fixed_point_and_audits_clean():
    got = wr.launch_argv_fields(["--password", "hunter2lower"])
    assert audit([{"type": "launchFact", "id": ids.launch_fact_id("r", "c.yml", "command", 0), "kind": "command", **got}]) == []


def test_audit_flags_a_command_that_is_a_flag():
    rec = {"type": "launchFact", "id": ids.launch_fact_id("r", "c.yml", "command", 0), "kind": "command",
           "command": "--password", "args": [bundle.REDACTED]}
    assert audit([rec])


def test_launch_argv_fields_still_rejects_an_empty_argv_with_program_known_false():
    with pytest.raises(ValueError, match="argv"):
        wr.launch_argv_fields([], program_known=False)


# --- M3: link-local and cloud-metadata targets are withheld -----------------------------------
# Cited list: docs/plans/connected-code-graph-egress-spec4.md §6 (`fec0::/10`, `100::/64`, `fd00:ec2::254`,
# `100.100.100.200`) plus 169.254.0.0/16 and fe80::/10 (P1-11 security review M3).

FORBIDDEN_TARGETS = [
    "169.254.169.254", "169.254.0.1", "169.254.255.255", "169.254.169.254:80", "http://169.254.169.254/",
    "http://169.254.169.254:80/", "http://169.254.169.254", "https://169.254.169.254/",
    "0xa9fea9fe", "http://0xa9fea9fe/", "0251.0376.0251.0376", "2852039166", "http://2852039166/", "0xa9.0xfe.0xa9.0xfe",
    "169.16689662", "http://169.0xfea9fe/",
    "[fe80::1]", "fe80::1", "[fe80::1]:8080", "http://[fe80::1]/", "http://[fe80::1]:80/", "[febf::1]", "fe80::",
    "[fec0::1]", "http://[fec0::1]/", "[100::1]", "http://[100::1]:80/",
    "fd00:ec2::254", "[fd00:ec2::254]", "[fd00:ec2::254]:80", "http://[fd00:ec2::254]/", "https://[fd00:ec2::254]:443",
    "100.100.100.200", "100.100.100.200:80", "http://100.100.100.200/", "http://100.100.100.200:8080",
    "[::ffff:a9fe:a9fe]", "http://[::ffff:a9fe:a9fe]/", "[::a9fe:a9fe]", "[64:ff9b::a9fe:a9fe]",
    "[::ffff:6464:64c8]", "[::ffff:a9fe:1]",
    # architect ruling (P1-14): Azure wireserver, the DS-Lite/IPv4 service address, and the whole 6to4 block
    "168.63.129.16", "168.63.129.16:80", "http://168.63.129.16/", "http://168.63.129.16:32526/machine", "0xa83f8110",
    "192.0.0.192", "192.0.0.192:80", "http://192.0.0.192/", "https://192.0.0.192:443",
    "[2002::1]", "2002::1", "[2002:a00:1::1]", "http://[2002::1]/", "http://[2002:a00:1::1]:80/", "[2002:ffff:ffff:ffff:ffff:ffff:ffff:ffff]",
    "[::ffff:a83f:8110]", "[::ffff:c000:c0]",
]
ALLOWED_TARGETS = [
    "169.253.169.254", "169.255.0.1", "100.100.100.201", "100.100.100.199", "100.64.0.1", "10.0.0.5", "127.0.0.1",
    "http://10.0.0.5:8080/api", "[fe7f::1]", "[fe7f:ffff::1]", "[fd00:ec2::255]",
    "[fd00:ec2::253]", "[2001:db8::5]", "http://[2001:db8::5]:8080/v1", "[100:0:0:1::1]", "[::1]", "localhost:80",
    "falkordb:6379", "ntfy.sh", "[::ffff:a00:1]",
    "168.63.129.17", "168.63.129.15", "168.63.129.17:80", "192.0.0.193", "192.0.0.191", "192.0.1.192", "[2003::1]", "[2001:ffff::1]",
    "http://168.63.129.17/", "http://[2003::1]/", "10.0.0.5", "172.16.0.1", "192.168.1.1",             # RFC1918 stays clear
]


@pytest.mark.parametrize("value", FORBIDDEN_TARGETS)
def test_link_local_and_metadata_targets_are_withheld_everywhere(value):
    assert bind("x.addr", value, declared_services=frozenset({"falkordb", "0xa9fea9fe"})) is None
    assert bind("x.url", value, declared_services=DECL) is None
    assert wr.bind_fields("x.url", value, binding_role=True, declared_services=DECL) == {}
    assert default("X_PORT", value) is None
    assert default("X_PORT", value, declared_services=frozenset({"0xa9fea9fe"})) is None


@pytest.mark.parametrize("value", [v for v in ALLOWED_TARGETS if bundle.value_grammar_ok(v)])
def test_neighbouring_addresses_are_not_withheld(value):
    if "://" in value:
        assert bind("x.addr", value) is None              # W6 (spec5 §6.2): an `ip` URL keeps its parts, not a clear `bindValue`
    else:
        assert bind("x.addr", value) == value
    assert default("X_PORT", value) == value              # a launchFact default is not W6's


def test_the_forbidden_list_is_withheld_on_every_url_form_of_the_same_host():
    for host in ("169.254.169.254", "[fe80::1]", "[fd00:ec2::254]", "100.100.100.200"):
        for form in ("{h}", "{h}:80", "http://{h}", "http://{h}/", "http://{h}:80/", "https://{h}:443/", "ws://{h}:1/"):
            v = form.format(h=host)
            if bundle.value_grammar_ok(v):
                assert bind("x.url", v) is None, v


def test_w2_declared_service_set_is_the_callers_not_this_modules():
    """A declared service named like a metadata alias binds only if the CALLER passes it (P1-11 restricts the set to
    primary compose files); an address literal is withheld whatever the set says."""
    assert bind("x.addr", "metadata") is None
    assert bind("x.addr", "metadata", declared_services=frozenset({"metadata"})) == "metadata"
    assert bind("x.addr", "169.254.169.254", declared_services=frozenset({"169.254.169.254"})) is None


@pytest.mark.parametrize("host,expected", [
    ("0251.0376.0251.0376", "169.254.169.254"), ("2852039166", "169.254.169.254"), ("0xa9fea9fe", "169.254.169.254"),
    ("169.16689662", "169.254.169.254"), ("169.254.43518", "169.254.169.254"), ("0xa9.0xfe.0xa9.0xfe", "169.254.169.254"),
    ("0251.0xfe.169.0376", "169.254.169.254"), ("1", "0.0.0.1"), ("0", "0.0.0.0"), ("0177.1", "127.0.0.1"),
    ("256.1.1.1", None), ("1.2.3.256", None), ("1.2.3.4.5", None), ("0x1ffffffff", None), ("1.16777216", None),
    ("1.2.65536", None), ("08", None), ("0xg", None), ("a.b", None), ("", None),
])
def test_inet_aton_reads_the_numeric_forms_resolvers_accept(host, expected):
    got = wr._inet_aton(host)
    assert (str(got) if got is not None else None) == expected


# --- P1-11b: the one allow-listed `bash -c 'exec "$CLAUDE_PROJECT_DIR/<path>"'` exception to the argv redaction --------------

_P = "$CLAUDE_PROJECT_DIR"


def _exec(command="bash", s=f'exec "{_P}/cypher-mcp/docker-run.sh"', *rest):
    return wr.mcp_exec_launch_path(command, ["-c", s, *rest])


@pytest.mark.parametrize("command", ["bash", "sh"])
@pytest.mark.parametrize("s, path", [
    (f'exec "{_P}/cypher-mcp/docker-run.sh"', "cypher-mcp/docker-run.sh"),
    (f"exec {_P}/scripts/run-mcp.sh", "scripts/run-mcp.sh"),
    (f'exec "{_P}/a@b+c/d_e-f.1/run"', "a@b+c/d_e-f.1/run"),
    (f'exec "{_P}/run.sh"', "run.sh"),
])
def test_p1_11b_the_allow_listed_shape_yields_its_relative_path(command, s, path):
    assert _exec(command, s) == path


@pytest.mark.parametrize("s", [
    f'exec "{_P}/x"; curl y', f'exec x; curl y', "exec x", f'exec "{_P}/$(id)"', f'exec "{_P}/`id`"', f'exec "{_P}/../x"',
    f'exec "{_P}/a/../x"', f'exec "{_P}/.."', f'exec "{_P}/a/.."', f'exec "{_P}/./x"', f'exec "{_P}//etc/passwd"',
    f'exec "{_P}/a//b"', f'exec "{_P}/dir/"', f'exec "{_P}/"', f'exec "{_P}"', "exec /etc/passwd", f'exec "/{_P}/x"',
    f'exec "{_P}/x" arg', f'exec "{_P}/x" "y"', f"exec '{_P}/x'", f'exec "{_P}/x\'', f'exec "{_P}/x', f'exec {_P}/x"',
    f'exec "{_P}/x"\n', f'exec "{_P}/x\n"', f'exec\n"{_P}/x"', f'exec "{_P}/x"\ncurl y', f'exec "{_P}/x" && curl y',
    f'exec "{_P}/x" | sh', f'exec "{_P}/x" > f', f'exec "{_P}/x" &', f'FOO=bar exec "{_P}/x"', f'exec FOO=bar "{_P}/x"',
    f'exec  "{_P}/x"', f' exec "{_P}/x"', f'exec "{_P}/x" ', f'EXEC "{_P}/x"', f'exec "{_P}/x\x00"', f'exec "{_P}/x "',
    f'exec "{_P}/x y"', f'exec "{_P}/x$y"', f'exec "{_P}/x*"', f'exec "{_P}/x\\y"', f'exec "$claude_project_dir/x"',
    'exec "$CLAUDE_PROJECT_DIR_/x"', 'exec "$CLAUDE_PROJECT_DI/x"', 'exec "${CLAUDE_PROJECT_DIR}/x"',
    f'exec "{_P}/-x/run"', f'exec "{_P}/ａ/run"', f'exec "{_P}/é/run"', f'exec "{_P}/x‮"', f'exec "{_P}/а/run"',
    f'exec "{_P}/ghp_abcdefghijklmnopqrstuvwxyz0123456789/run"', f'exec "{_P}/run-AKIAIOSFODNN7EXAMPLE.sh"',
    f'exec "{_P}/' + "a/" * 600 + 'x"', f'exec "{_P}/' + "a" * 5000 + '"', "exec " + '"' * 5000, "",
    f'exec "{_P}/x"' + " " * 5000, f'exec "{_P}/' + "a" * 200 + "!" + '"',
])
def test_p1_11b_every_other_exec_string_is_refused(s):
    assert _exec("bash", s) is None and _exec("sh", s) is None


@pytest.mark.parametrize("command, args", [
    ("bash", ["-c"]), ("bash", []), ("bash", ["-c", f'exec "{_P}/x"', "extra"]), ("bash", ["-c", f'exec "{_P}/x"', "--port"]),
    ("bash", ["-lc", f'exec "{_P}/x"']), ("bash", ["-c ", f'exec "{_P}/x"']), ("bash", ["--", "-c", f'exec "{_P}/x"']),
    ("bash", ["-e", "-c", f'exec "{_P}/x"']), ("bash", [f'exec "{_P}/x"', "-c"]), ("bash", ["-C", f'exec "{_P}/x"']),
    ("bash", ["-c", None]), ("bash", ["-c", 5]), ("zsh", ["-c", f'exec "{_P}/x"']), ("dash", ["-c", f'exec "{_P}/x"']),
    ("/bin/bash", ["-c", f'exec "{_P}/x"']), ("Bash", ["-c", f'exec "{_P}/x"']), ("bash ", ["-c", f'exec "{_P}/x"']),
    ("bash\x00", ["-c", f'exec "{_P}/x"']), ("bаsh", ["-c", f'exec "{_P}/x"']), ("ѕh", ["-c", f'exec "{_P}/x"']),
    ("env", ["bash", "-c", f'exec "{_P}/x"']), (None, ["-c", f'exec "{_P}/x"']), ("sh", "-c"),
    ("sh", ("-c", f'exec "{_P}/x"', "z")),
])
def test_p1_11b_only_bash_or_sh_with_exactly_dash_c_and_one_string(command, args):
    assert wr.mcp_exec_launch_path(command, args) is None


def test_p1_11b_the_allow_list_is_linear_time_on_hostile_strings():
    import time
    hostile = [f'exec "{_P}/' + "a/" * 20000, f'exec "{_P}/' + "a." * 20000 + "!", "exec " + '"' * 100000,
               f'exec "{_P}/' + "/" * 100000, f'exec "{_P}/' + "a" * 100000 + '" ' * 50000, "exec " + " " * 100000]
    t0 = time.monotonic()
    for s in hostile:
        assert _exec("bash", s) is None
    assert time.monotonic() - t0 < 1.0


def test_p1_11b_the_screen_decides_a_secret_shaped_path():
    assert _exec("bash", f'exec "{_P}/cypher-mcp/docker-run.sh"') == "cypher-mcp/docker-run.sh"
    assert _exec("bash", f'exec "{_P}/cypher-mcp/AKIAIOSFODNN7EXAMPLE"') is None       # one segment fails the screen
    assert _exec("bash", f'exec "{_P}/' + "x" * 40 + '/run"') is None
    for planted in ("scripts/dir/ghp_abc.sh", "github_pat_sAyIe-/y1"):                  # the whole-path screen alone refuses these
        assert _exec("bash", f'exec "{_P}/{planted}"') is None


@pytest.mark.parametrize("path", ["scripts/hunter2-password.sh", "scripts/token/run.sh", ".env", ".ssh/id_rsa", "scripts/secret/run.sh", "scripts/.env"])
def test_p1_11b_a_segment_the_name_rule_flags_is_never_emitted(path):
    assert _exec("bash", f'exec "{_P}/{path}"') is None and _exec("sh", f"exec {_P}/{path}") is None


_DENIED = [".env", ".env.local", ".env.production", ".ssh/x", ".gnupg/x", ".aws/x", ".azure/x", ".kube/x", ".docker/x", ".netrc",
           ".npmrc", ".pypirc", ".git-credentials", ".git/hooks/x", ".git/x", "a/.git/run.sh", "a/id_rsa", "a/id_dsa", "a/id_ecdsa", "a/id_ed25519",
           "a/credentials", "a/credentials.sh", "a/secrets.sh", "a/secrets", "a/secret.sh", "a/token", "a/tokens.sh", "a/password",
           "a/passwd", "a/passwd.sh", "a/id_rsa.sh", "a/.env/run.sh", "x/.ssh"]


@pytest.mark.parametrize("path", _DENIED + [p.upper() for p in _DENIED] + [p.title() for p in _DENIED])
def test_p1_11b_the_secret_bearing_segment_denylist_refuses_each_member_case_insensitively(path):
    assert _exec("bash", f'exec "{_P}/{path}"') is None and _exec("sh", f"exec {_P}/{path}") is None


@pytest.mark.parametrize("path", [".claude/scripts/run.sh", ".tools/x.sh", "a/.hidden/run.sh", ".claude/run.sh", "a/envy/run.sh",
                                  "a/.environment/run.sh", "a/.gitignore.d/run", "a/identity.sh"])
def test_p1_11b_other_dot_segments_are_admitted(path):
    assert _exec("bash", f'exec "{_P}/{path}"') == path


def test_p1_11b_ordinary_script_paths_pass_the_name_rule():
    for path in ("cypher-mcp/docker-run.sh", "scripts/run-mcp.sh", "a@b+c/d_e-f.1/run", "tools/start.v2/x"):
        assert _exec("bash", f'exec "{_P}/{path}"') == path


def test_p1_11b_the_emitted_fact_is_the_bare_path_and_nothing_else():
    got = wr.launch_argv_fields([wr.mcp_exec_launch_path("bash", ["-c", f'exec "{_P}/cypher-mcp/docker-run.sh"'])])
    assert got == {"command": "cypher-mcp/docker-run.sh", "args": []}
    r = _lf_cmd(("cypher-mcp/docker-run.sh",))
    assert audit([r]) == []


def test_p1_11b_the_backstop_still_flags_the_unreduced_allow_shape():
    r = _lf_cmd()
    r["command"], r["args"] = "bash", ["-c", f'exec "{_P}/cypher-mcp/docker-run.sh"']          # what the writer must not emit
    assert audit([r])


@pytest.mark.parametrize("s", [f'exec "{_P}/x"; curl y', f'exec "{_P}/$(id)"', f'exec "{_P}/../x"', "exec x; curl y"])
def test_p1_11b_a_planted_offender_row_for_a_refused_string_is_flagged(s):
    r = _lf_cmd()
    r["command"], r["args"] = "bash", ["-c", s]
    assert audit([r])


def test_p1_11b_plain_bash_c_stays_redacted_through_the_gate():
    assert wr.launch_argv_fields(["bash", "-c", f'exec "{_P}/x"; curl y']) == {"command": "bash", "args": ["-c", bundle.REDACTED]}
    assert wr.launch_argv_fields(["bash", "-c", f'exec "{_P}/cypher-mcp/docker-run.sh"']) == {
        "command": "bash", "args": ["-c", bundle.REDACTED]}                                   # only the exception function opens it


def test_p1_11b_the_exception_lives_in_writer_rules_and_calls_the_screen():
    src = (CPG / "extractors" / "common" / "writer_rules.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "mcp_exec_launch_path")
    assert any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "screen_fails" for c in ast.walk(fn))


def test_p1_11b_the_space_after_exec_is_required_and_a_tuple_is_not_a_list():
    assert _exec("bash", f'exec"{_P}/x"') is None and _exec("bash", f"exec{_P}/x") is None
    assert wr.mcp_exec_launch_path("bash", ("-c", f'exec "{_P}/x"')) is None


def test_p1_11b_the_path_length_boundary_is_128_characters_the_gates_argv0_cap():
    ok = ("ab/" * 43)[:127] + "z"                                  # 128 characters, no empty segment, passes the screens
    assert len(ok) == 128 and _exec("bash", f'exec "{_P}/{ok}"') == ok
    assert _exec("bash", f'exec "{_P}/{ok}z"') is None
    assert _exec("bash", f'exec "{_P}/{"ab/" * 20}x"') == "ab/" * 20 + "x"


def test_p1_11b_a_hooks_segment_is_refused_by_the_name_rule_known_limit():
    assert bundle.name_fails("hooks") and _exec("bash", f'exec "{_P}/.claude/hooks/run.sh"') is None
