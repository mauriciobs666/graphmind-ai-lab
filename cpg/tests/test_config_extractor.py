"""P1-11 — the config, Compose, `.mcp.json`, Kiro and Dockerfile extractors (F-6, SA-3 allow-by-role).

Plan: `docs/plans/connected-code-graph-p1-extraction.md` § P1-11; spec: `-bundle-spec3.md` §1.6, §1.7, §5.17 (per-format
`keyPath`), `-bundle-spec5.md` §6.1 E3 and §9.2. Fixtures: `tests/data/config/repo1/` with its golden
`tests/data/config/expected/repo1.json` (rows: keyPath, line, text, bindValue-or-null; facts without ids and hashes). Every
writer test runs `writer_rules.audit_records(..., texts=<the raw texts>)` and asserts no findings. Test keys only; no network, no
Docker, no store.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import hmac
import json
import re
import signal
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import pytest

from ccg import atrest, bundle, ids, valuehash
from extractors.common import writer_rules as wr
from extractors.config import extract as X
from extractors.config import formats as F
from extractors.config import launch as L
from extractors.joern.normalize import assemble_bundle
from extractors.common.errors import SourceRefused

CPG = Path(__file__).resolve().parent.parent
DATA = CPG / "tests" / "data" / "config"
REPO_DIR = DATA / "repo1"
KEY = bytes(range(64, 96))
REPO = "r1"


def repo_files() -> dict[str, bytes]:
    return {str(p.relative_to(REPO_DIR)): p.read_bytes() for p in sorted(REPO_DIR.rglob("*")) if p.is_file()}


def golden() -> dict:
    return json.loads((DATA / "expected" / "repo1.json").read_text(encoding="utf-8"))


def run(files: dict[str, str | bytes], **kw) -> X.ConfigResult:
    return X.extract_config(REPO, {p: (v.encode("utf-8") if isinstance(v, str) else v) for p, v in files.items()}, key=KEY, **kw)


def entries(res: X.ConfigResult, path: str) -> dict[str, dict]:
    return {r["keyPath"]: r for r in res.records if r["type"] == "configEntry" and r["path"] == path}


def facts(res: X.ConfigResult, path: str, kind: str | None = None) -> list[dict]:
    return [r for r in res.records if r["type"] == "launchFact" and r["path"] == path and (kind is None or r["kind"] == kind)]


def binds(res: X.ConfigResult, path: str) -> dict[str, str]:
    return {k: r["bindValue"] for k, r in entries(res, path).items() if "bindValue" in r}


@pytest.fixture(scope="module")
def repo1() -> X.ConfigResult:
    return X.extract_config(REPO, repo_files(), key=KEY)


# --- goldens, one fixture file per format --------------------------------------------------------------------

_GOLDEN_PATHS = sorted(golden())


@pytest.mark.parametrize("path", _GOLDEN_PATHS)
def test_golden_entries_and_facts_per_file(repo1, path):
    want = golden()[path]
    got_entries = sorted(([r["keyPath"], r["line"], r.get("bindValue")] for r in entries(repo1, path).values()),
                         key=lambda e: (e[1], e[0]))
    assert got_entries == sorted(([k, ln, b] for k, ln, _t, b in want["entries"]), key=lambda e: (e[1], e[0]))
    got_facts = [{k: v for k, v in r.items() if k not in ("id", "path", "defaultHash")} | {"hasDefaultHash": "defaultHash" in r}
                 for r in facts(repo1, path)]
    key = lambda f: (f["kind"], f["ordinal"])           # noqa: E731
    assert sorted(got_facts, key=key) == sorted(want["facts"], key=key)


def test_the_golden_covers_every_format_of_5_17_and_the_table_has_one_row_each():
    spec = (CPG.parent / "docs" / "plans" / "connected-code-graph-bundle-spec3.md").read_text(encoding="utf-8")
    row = next(ln for ln in spec.splitlines() if ln.startswith("| `format` |"))
    in_spec = set(re.findall(r"`([a-z-]+)`", row.split("| yes |")[0].split("| `format` |")[1]))
    assert in_spec == {"json", "yaml", "toml", "compose", "env", "dockerfile", "mcp-json", "kiro-agent", "ini"}
    assert [r.name for r in F.FORMATS if r.name in in_spec] == [r.name for r in F.FORMATS]
    assert sorted(r.name for r in F.FORMATS) == sorted(in_spec)                      # exactly one row per format
    assert all(len(r.derivation) > 20 for r in F.FORMATS)
    seen = {r["format"] for r in X.extract_config(REPO, repo_files(), key=KEY).records if r["type"] == "configEntry"}
    assert seen == in_spec                                                           # every row has a test input


def test_classification_is_by_specific_name_before_extension():
    c = lambda p: F.classify(p).name                                                 # noqa: E731
    assert c(".mcp.json") == "mcp-json" and c("a/.mcp.example.json") == "mcp-json"
    assert c("kiro/.kiro/agents/x.json") == "kiro-agent" and c("a/.kiro/other.json") == "json"
    assert c("falkor-chat/compose.yaml") == "compose" and c("docker-compose.override.yml") == "compose"
    assert c("deploy/compose.example.yaml") == "compose" and c("ci/pipeline.yaml") == "yaml"
    assert c("Dockerfile") == "dockerfile" and c("svc/Dockerfile.dev") == "dockerfile" and c("a/b.Dockerfile") == "dockerfile"
    assert c(".env") == "env" and c("x/.env.example") == "env" and c("prod.env") == "env"
    assert c("a.json") == "json" and c("a.yml") == "yaml" and c("a.toml") == "toml" and c("a.ini") == "ini" and c("a.cfg") == "ini"
    assert F.classify("a.py") is None and F.classify("README.md") is None


# --- the role rule (E3) ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("segment", ["port", "PORT", "FALKORDB_PORT", "base_url", "baseUrl", "BASE-URL", "HTTPServer", "apiHost",
                                     "listen", "bind", "endpoint", "uri", "addr", "address", "hostname", "x.host", "mcp_server"])
def test_a_role_token_in_the_last_segment_is_a_binding_role(segment):
    assert F.binding_role("json", ("a", segment))


@pytest.mark.parametrize("segment", ["timeout", "ports", "hosts", "reporting", "support", "database", "mcpServers", "imported",
                                     "baseline", "urls", "name", "image"])
def test_a_substring_of_a_role_word_is_not_a_role_token(segment):
    assert not F.binding_role("json", ("a", segment))


def test_the_last_segment_decides_not_the_first():
    assert not F.binding_role("json", ("port", "timeout"))          # first segment has the token, last does not
    assert F.binding_role("json", ("timeout", "port"))


def test_an_index_is_not_a_name_so_it_never_carries_a_generic_role():
    assert not F.binding_role("json", ("hosts", 0))
    assert not F.binding_role("json", ("url", 0))                   # the token sits one segment up: not the last segment


def test_camel_case_and_separators_split_tokens():
    assert F.role_tokens("HTTPServer") == ["http", "server"]
    assert F.role_tokens("baseURL") == ["base", "url"]
    assert F.role_tokens("a_b-c.d") == ["a", "b", "c", "d"]


@pytest.mark.parametrize("fmt,segs,want", [
    ("compose", ("services", "api", "ports", 0), True), ("compose", ("services", "api", "expose", 3), True),
    ("compose", ("services", "api", "volumes", 0), False), ("compose", ("services", "api", "ports"), False),
    ("compose", ("services", "api", "ports", 0, "target"), False), ("compose", ("services", "api", "ports", "x"), False), ("yaml", ("services", "api", "ports", 0), False),
    ("mcp-json", ("mcpServers", "s", "url"), True), ("kiro-agent", ("mcpServers", "s", "url"), True),
    ("mcp-json", ("mcpServers", "s", "args", 0), False), ("mcp-json", ("mcpServers", "s", "timeout"), False),
    ("mcp-json", ("other", "s", "x"), False),
])
def test_explicit_binding_rows(fmt, segs, want):
    assert F.binding_role(fmt, segs) is want


def test_no_binding_role_no_bind_value_even_for_a_bindable_value():
    r = run({"a.json": '{"timeout": "http://localhost:8000/mcp", "url": "http://localhost:8000/mcp"}'})
    assert binds(r, "a.json") == {"url": "http://localhost:8000/mcp"}              # BP-37: `x.timeout` has no role


def test_array_elements_under_args_command_entrypoint_are_never_bind_values():
    r = run({"a.json": json.dumps({"args": ["http://localhost:8000/mcp"], "command": ["8200"], "entrypoint": ["localhost:8000"],
                                   "x": {"args": ["http://localhost:1/"]}}),
             ".mcp.json": json.dumps({"mcpServers": {"s": {"command": "run", "args": ["http://localhost:8000/mcp", "8200"]}}}),
             "compose.yaml": "services:\n  a:\n    command: ['8200']\n    entrypoint: ['http://localhost:8000/']\n"})
    assert all(not binds(r, p) for p in ("a.json", ".mcp.json", "compose.yaml"))
    assert facts(r, ".mcp.json", "command")[0]["args"][0].startswith("http")        # the value belongs to the launchFact


# --- bindValue / default / argv / hashes come from writer_rules ----------------------------------------------

def test_every_bind_value_traces_to_bind_fields(monkeypatch):
    seen: list[tuple[str, str, str]] = []
    real = wr.bind_fields

    def spy(kpath, text, **kw):
        out = real(kpath, text, **kw)
        if out:
            seen.append((kpath, text, out["bindValue"]))
        return out
    monkeypatch.setattr(X.wr, "bind_fields", spy)
    res = X.extract_config(REPO, repo_files(), key=KEY)
    emitted = [(r["keyPath"], r["bindValue"]) for r in res.records if r["type"] == "configEntry" and "bindValue" in r]
    assert emitted and sorted(emitted) == sorted((k, b) for k, _t, b in seen)
    monkeypatch.setattr(X.wr, "bind_fields", lambda *a, **k: {})                     # withheld everywhere: nothing binds
    assert not any("bindValue" in r for r in X.extract_config(REPO, repo_files(), key=KEY).records)


def test_every_default_and_argv_traces_to_the_writer_rules(monkeypatch):
    monkeypatch.setattr(X.wr, "default_fields", lambda key, lid, env, text, **kw: {"defaultHash": "0" * 64})
    monkeypatch.setattr(X.wr, "launch_argv_fields", lambda argv, **kw: {"command": "C", "args": ["A"]})
    res = X.extract_config(REPO, repo_files(), key=KEY)
    envs = [r for r in res.records if r["type"] == "launchFact" and r["kind"] == "env-default"]
    cmds = [r for r in res.records if r["type"] == "launchFact" and r["kind"] == "command"]
    assert envs and all("default" not in r and r["defaultHash"] == "0" * 64 for r in envs)
    assert cmds and all(r["command"] == "C" and r["args"] == ["A"] for r in cmds)


def _independent_hash(rid: str, text: str) -> str:
    return hmac.new(KEY, rid.encode() + b"\x00" + unicodedata.normalize("NFC", text).encode(), hashlib.sha256).hexdigest()


def test_value_hash_is_hmac_over_the_entrys_own_id_computed_independently(repo1):
    for path, want in golden().items():
        got = entries(repo1, path)
        for kpath, _line, text, _bind in want["entries"]:
            rid = ids.config_entry_id(REPO, path, kpath)
            assert got[kpath]["id"] == rid
            assert got[kpath]["valueHash"] == _independent_hash(rid, text)


def test_every_value_hash_comes_from_value_hash_fields(monkeypatch):
    calls = []
    real = wr.value_hash_fields

    def spy(key, rid, text):
        calls.append(rid)
        return real(key, rid, text)
    monkeypatch.setattr(X.wr, "value_hash_fields", spy)
    res = X.extract_config(REPO, repo_files(), key=KEY)
    cfg = [r for r in res.records if r["type"] == "configEntry"]
    assert sorted(calls) == sorted(r["id"] for r in cfg)
    monkeypatch.setattr(X.wr, "value_hash_fields", lambda key, rid, text: {"valueHash": "0" * 64})
    assert all(r["valueHash"] == "0" * 64 for r in X.extract_config(REPO, repo_files(), key=KEY).records if r["type"] == "configEntry")


def test_the_same_value_under_two_key_paths_hashes_differently():
    r = run({"a.json": '{"a": {"port": "8200"}, "b": {"port": "8200"}}'})
    e = entries(r, "a.json")
    assert e["a.port"]["valueHash"] != e["b.port"]["valueHash"]
    assert e["a.port"]["bindValue"] == e["b.port"]["bindValue"] == "8200"


# the env-default texts of the fixture (a withheld default is hashed too; without its text the audit cannot check it)
_DEFAULT_TEXTS = {("compose.yaml", 0): "6379", ("compose.yaml", 1): "acme", ("examples/compose.example.yaml", 0): "6379",
                  ("examples/compose.example.yaml", 1): "acme", (".env", 0): "acme", (".env.example", 0): "acme",
                  ("Dockerfile", 0): "8300"}


def _audit_texts(res: X.ConfigResult) -> dict[str, str]:
    texts = {}
    for path, want in golden().items():
        for kpath, _l, text, _b in want["entries"]:
            texts[ids.config_entry_id(REPO, path, kpath)] = text
    for r in res.records:
        if r["type"] == "launchFact" and r["kind"] == "env-default":
            texts[r["id"]] = _DEFAULT_TEXTS[(r["path"], r["ordinal"])]
    return texts


def test_audit_records_finds_nothing_in_the_fixture_repo(repo1):
    texts = _audit_texts(repo1)
    assert sum(1 for r in repo1.records if r["id"] in texts) == len(repo1.records) - sum(
        1 for r in repo1.records if r["type"] == "launchFact" and r["kind"] != "env-default")
    assert wr.audit_records(repo1.records, key=KEY, declared_services=repo1.declared_services, texts=texts) == []


def test_audit_catches_a_wrong_domain_hash_on_a_withheld_default(repo1):
    """The done-condition's reason for `texts`: a hash in the wrong domain on a withheld value passes without it."""
    r = next(r for r in repo1.records if r["type"] == "launchFact" and r["kind"] == "env-default" and "default" not in r)
    bad = dict(r, defaultHash=wr.default_hash(KEY, ids.launch_fact_id(REPO, "x", "env-default", 9), "acme"))
    texts = _audit_texts(repo1)
    recs = [x if x is not r else bad for x in repo1.records]
    assert wr.audit_records(recs, key=KEY, declared_services=repo1.declared_services) == []         # silent without texts
    assert wr.audit_records(recs, key=KEY, declared_services=repo1.declared_services, texts=texts)


def test_withheld_fields_are_absent_never_null(repo1):
    for r in repo1.records:
        assert all(v is not None for v in r.values()), r


def test_env_default_present_only_for_a_role_named_variable(repo1):
    envs = {(r["envVar"], r.get("default")) for r in facts(repo1, "compose.yaml", "env-default")}
    assert envs == {("DB_PORT", "6379"), ("WS_ID", None)}                           # WS_ID has no role token: hash only
    assert all("defaultHash" in r for r in facts(repo1, "compose.yaml", "env-default"))


def test_a_default_that_passes_the_grammar_is_still_withheld_without_a_role_named_variable():
    r = run({"compose.yaml": "services:\n  a:\n    environment:\n      LOG_LEVEL: ${LOG_LEVEL:-8200}\n      A_PORT: ${A_PORT:-8200}\n"})
    f = {x["envVar"]: x for x in facts(r, "compose.yaml", "env-default")}
    assert "default" not in f["LOG_LEVEL"] and f["A_PORT"]["default"] == "8200"


def test_w2_a_declared_service_default_is_kept_and_an_undeclared_one_is_withheld():
    txt = "services:\n  acme:\n    image: x\n    environment:\n      A_HOST: ${A_HOST:-acme}\n      B_HOST: ${B_HOST:-other}\n"
    f = {r["envVar"]: r for r in facts(run({"compose.yaml": txt}), "compose.yaml", "env-default")}
    assert f["A_HOST"]["default"] == "acme" and "default" not in f["B_HOST"]
    assert "defaultHash" in f["B_HOST"]


def test_w2_binding_to_a_service_declared_in_another_file_counts_and_an_extra_set_adds_to_it():
    web = "services:\n  web:\n    environment:\n      API_HOST: api\n      PEER_HOST: peer\n"
    api = "services:\n  api:\n    image: x\n"
    alone = run({"web.compose.yaml": web})
    assert binds(alone, "web.compose.yaml") == {}
    both = run({"web.compose.yaml": web, "compose.yaml": api})
    assert binds(both, "web.compose.yaml") == {"services.web.environment.API_HOST": "api"}
    assert binds(run({"web.compose.yaml": web}, extra_declared_services=["peer"]), "web.compose.yaml") == {
        "services.web.environment.PEER_HOST": "peer"}


def test_an_env_default_under_an_alias_is_one_fact_not_one_per_occurrence():
    r = run({"compose.yaml": "x-env: &e {A_PORT: '${A_PORT:-8200}'}\nservices:\n  a: {environment: *e}\n  b: {environment: *e}\n"})
    assert [(f["envVar"], f["default"]) for f in facts(r, "compose.yaml", "env-default")] == [("A_PORT", "8200")]
    assert binds(r, "compose.yaml") == {}                               # an interpolated value never binds, alias or not


def test_env_default_forms():
    d = L.env_defaults("a ${A_PORT:-8200} ${B_PORT-9} ${C:-} ${D_PORT:-${E}} $F ${G:=1}", 7)
    assert [(x.env_var, x.default_text, x.line) for x in d] == [("A_PORT", "8200", 7), ("B_PORT", "9", 7), ("C", "", 7),
                                                                 ("D_PORT", "${E", 7)]
    assert L.env_defaults("x" * 10, 1) == []
    assert len(L.env_defaults("${A:-1}" * 40, 1)) == L.MAX_ENV_PER_TEXT


def test_interpolated_values_never_bind_and_secrets_are_never_stored(repo1):
    assert "services.falkordb.ports[0]" not in binds(repo1, "compose.yaml")
    blob = json.dumps(repo1.records) + json.dumps(repo1.report)
    for secret in ("hunter2", "t0ps3cret", "abc123"):
        assert secret not in blob
    assert "AKIA" not in blob


def test_a_secret_flag_value_in_an_argv_is_redacted_by_the_gate():
    r = run({".mcp.json": json.dumps({"mcpServers": {"s": {"command": "redis-cli", "args": ["AUTH", "hunter2", "--password", "hunter2",
                                                                                          "--port", "6379"]}}})})
    got = facts(r, ".mcp.json")[0]
    assert "hunter2" not in json.dumps(got) and got["args"][-2:] == ["--port", "6379"]
    assert [got["command"], *got["args"]] == bundle.gate_argv([got["command"], *got["args"]])


# --- compose ---------------------------------------------------------------------------------------------------

def test_compose_list_environment_splits_on_the_first_equals_and_drops_bare_names():
    r = run({"compose.yaml": "services:\n  a:\n    environment:\n      - X_URL=http://localhost:1/\n      - Y=a=b\n      - BARE\n"})
    e = entries(r, "compose.yaml")
    assert set(e) == {"services.a.environment.X_URL", "services.a.environment.Y"}
    assert binds(r, "compose.yaml") == {"services.a.environment.X_URL": "http://localhost:1/"}
    assert e["services.a.environment.Y"]["valueHash"] == _independent_hash(e["services.a.environment.Y"]["id"], "a=b")


@pytest.mark.parametrize("spec,want", [
    ("8000", [(8000, "tcp")]), ("8200:8000", [(8200, "tcp"), (8000, "tcp")]), ("8000:8000", [(8000, "tcp")]),
    ("127.0.0.1:3000:3000/udp", [(3000, "udp")]), ("127.0.0.1:3001:3000", [(3001, "tcp"), (3000, "tcp")]),
    ("${P:-1}:6379", [(6379, "tcp")]), ("8000-8010:8000-8010", []), ("0:80", [(80, "tcp")]), ("65536", []), ("x", []),
    ("[::1]:8080:80", [(8080, "tcp"), (80, "tcp")]), ("80/weird", [(80, "tcp")]), ("08000", []),
])
def test_compose_port_forms(spec, want):
    assert L.parse_port_spec(spec) == want


def test_compose_build_context_is_recorded_relative_and_only_when_it_is_a_plain_path():
    ok = run({"compose.yaml": "services:\n  a:\n    build:\n      context: ../../fx-job\n  b:\n    build: ./b\n"})
    assert [(f["service"], f["mountPath"]) for f in facts(ok, "compose.yaml", "mount")] == [("a", "../../fx-job"), ("b", "./b")]
    for bad in ("https://user:pw@github.com/x/y.git", "${CTX}", "git@github.com:x/y.git", "a b", ""):
        r = run({"compose.yaml": f"services:\n  a:\n    build:\n      context: {json.dumps(bad)}\n"})
        assert facts(r, "compose.yaml", "mount") == [], bad


def test_compose_long_port_syntax_yields_published_then_target():
    r = run({"compose.yaml": "services:\n  a:\n    ports:\n      - {target: 80, published: 8080, protocol: udp}\n"})
    assert [(f["port"], f["protocol"]) for f in facts(r, "compose.yaml", "port")] == [(8080, "udp"), (80, "udp")]


def test_compose_command_string_is_split_not_expanded():
    r = run({"compose.yaml": "services:\n  a:\n    command: python run.py --port 8000 $HOME\n  b:\n    command: \"unbalanced 'quote\"\n"})
    f = facts(r, "compose.yaml", "command")
    assert len(f) == 1 and f[0]["service"] == "a" and f[0]["command"] == "python"
    assert f[0]["args"] == bundle.gate_argv(["python", "run.py", "--port", "8000", "$HOME"])[1:] and f[0]["args"][:3] == ["run.py", "--port", "8000"]


# --- .mcp.json and Kiro ----------------------------------------------------------------------------------------

def test_mcp_json_yields_the_launch_chain_input_and_binds_only_url_and_role_named_env(repo1):
    f = facts(repo1, ".mcp.json")
    assert len(f) == 1 and f[0]["kind"] == "command" and f[0]["service"] == "cypher" and f[0]["origin"] == "explicit"
    assert (f[0]["command"], f[0]["args"]) == ("bash", ["-c", "run.sh", "--port", "8200"])
    assert binds(repo1, ".mcp.json") == {"mcpServers.cypher.env.FALKORDB_HOST": "host.docker.internal",
                                         "mcpServers.cypher.env.FALKORDB_PORT": "6379",
                                         "mcpServers.team.url": "http://localhost:8200/mcp"}


def test_kiro_agent_binds_its_mcp_url_only(repo1):
    assert binds(repo1, ".kiro/agents/demo.json") == {"mcpServers.falkor-chat.url": "http://localhost:8000/mcp"}


def test_server_names_that_fail_the_screen_are_not_written_as_the_service():
    name = "x7k2-q9w4-m1p8-z3v6"
    r = run({".mcp.json": json.dumps({"mcpServers": {name: {"command": "run"}, "ok": {"command": "run"}}})})
    got = facts(r, ".mcp.json")
    assert sorted("service" in f for f in got) == [False, True]
    assert name not in json.dumps(facts(r, ".mcp.json"))


# --- Dockerfile ------------------------------------------------------------------------------------------------

def test_dockerfile_reads_the_final_stage_only_and_the_last_cmd_and_entrypoint(repo1):
    e = entries(repo1, "Dockerfile")
    assert "ENV.BUILD_ONLY" not in e and "ENV.APP_PORT" in e and "ARG.BASE_PORT" in e
    cmds = facts(repo1, "Dockerfile", "command")
    assert [(f["command"]) for f in cmds] == ["tini", "<redacted>"] and [f["ordinal"] for f in cmds] == [0, 1]
    assert [(f["port"], f["protocol"]) for f in facts(repo1, "Dockerfile", "port")] == [(8300, "tcp"), (9300, "tcp")]


def test_dockerfile_argv_forms():
    ins = F.dockerfile_instructions
    assert F.dockerfile_argv(ins('CMD ["a", "b c"]')[0]) == ["a", "b c"]
    assert F.dockerfile_argv(ins("CMD a 'b c' --x=1")[0]) == ["a", "b c", "--x=1"]
    assert F.dockerfile_argv(ins("CMD [not json")[0]) == ["[not", "json"]
    assert F.dockerfile_argv(ins("CMD unbalanced 'quote")[0]) is None
    assert F.dockerfile_argv(None) is None
    assert F.dockerfile_argv(ins('CMD [1, "a"]')[0]) == ["[1,", "a]"]               # a non-string array is shell form, as Docker reads it


def test_dockerfile_continuations_comments_and_legacy_env():
    txt = "FROM x\n# c\nENV A 1\nENV B=2 \\\n  # inner comment\n  C=\"d e\"\nEXPOSE 80 443/udp $P abc\nARG NOPE\nARG YES=1 ALSO=2\n"
    r = run({"Dockerfile": txt})
    e = {k: v["line"] for k, v in entries(r, "Dockerfile").items()}
    assert e == {"ENV.A": 3, "ENV.B": 4, "ENV.C": 4, "EXPOSE[0]": 7, "EXPOSE[1]": 7, "EXPOSE[2]": 7, "EXPOSE[3]": 7,
                 "ARG.YES": 9, "ARG.ALSO": 9}
    assert [(f["port"], f["protocol"]) for f in facts(r, "Dockerfile", "port")] == [(80, "tcp"), (443, "udp")]


def test_the_last_cmd_and_the_last_entrypoint_of_the_final_stage_win():
    r = run({"Dockerfile": 'FROM a\nCMD ["one"]\nENTRYPOINT ["e1"]\nCMD ["two"]\nENTRYPOINT ["e2", "x"]\n'})
    assert [(f["command"], f["args"]) for f in facts(r, "Dockerfile", "command")] == [("<redacted>", ["two"]), ("e2", ["x"])]
    assert {"command[0]", "entrypoint[0]", "entrypoint[1]"} == set(entries(r, "Dockerfile"))


def test_a_dockerfile_with_no_from_still_reads():
    r = run({"Dockerfile": "ENV X_PORT=1\nCMD [\"run\"]\n"})
    assert "ENV.X_PORT" in entries(r, "Dockerfile") and facts(r, "Dockerfile", "command")[0]["command"] == "run"


def test_uvicorn_port_flag_is_a_command_launch_fact_with_the_flag_in_args(repo1):
    c = next(f for f in facts(repo1, "compose.yaml", "command") if f["command"] == "uvicorn")
    assert c["args"] == ["app:app", "--host", "0.0.0.0", "--port", "8000"]


# --- ordinals, lines, ids, schema, the whole bundle ------------------------------------------------------------

def test_ordinals_are_dense_per_path_and_kind_and_lines_do_not_decrease(repo1):
    groups: dict[tuple, list] = {}
    for r in repo1.records:
        if r["type"] == "launchFact":
            groups.setdefault((r["path"], r["kind"]), []).append(r)
    assert groups
    for g in groups.values():
        g.sort(key=lambda r: r["ordinal"])
        assert [r["ordinal"] for r in g] == list(range(len(g)))
        assert [r["line"] for r in g] == sorted(r["line"] for r in g)


def test_ids_are_unique_and_recomputable(repo1):
    assert len({r["id"] for r in repo1.records}) == len(repo1.records)
    for r in repo1.records:
        want = (ids.config_entry_id(REPO, r["path"], r["keyPath"]) if r["type"] == "configEntry"
                else ids.launch_fact_id(REPO, r["path"], r["kind"], r["ordinal"]))
        assert r["id"] == want


def _bundle_for(res: X.ConfigResult, path: str) -> list[dict]:
    """A whole bundle for one config unit (rootPath is the file), through the validator."""
    uid = ids.unit_id(REPO, path, "config")
    recs = [r for r in res.records if r["path"] == path]
    by_type = {t: sorted((r for r in recs if r["type"] == t), key=lambda r: r["id"]) for t in ("configEntry", "launchFact")}
    file_rec = {"type": "file", "id": ids.file_id(REPO, path), "repoId": REPO, "unitId": uid, "path": path, "lang": "config",
                "kind": "config", "role": res.roles[path], "contentHash": "0" * 64, "lineCount": 1}
    counts = {t: 0 for t in bundle.RECORD_TYPES[1:]} | {"file": 1, "configEntry": len(by_type["configEntry"]),
                                                       "launchFact": len(by_type["launchFact"])}
    head = {"type": "header", "bundleSchemaVersion": "1.0", "unit": {"repoId": REPO, "rootPath": path, "lang": "config"},
            "unitId": uid, "builtCommit": "1" * 40, "builtTree": "2" * 40, "builtAt": "2026-10-29T12:00:00Z",
            "tools": {"joern": "4.0.579", "frontend": "none", "frontendVersion": "0", "normalizer": "config",
                      "normalizerVersion": "0.0.1", "extractScript": "none", "extractScriptHash": "0" * 64},
            "sandboxAuditIds": ["0f8fad5b-d9cb-469f-a165-70867728950e"], "egressMode": "offline", "flowProjected": False,
            "fieldDepthCap": 3, "vendoredDirs": [], "history": "available", "counts": counts}
    head.update(wr.bind_header_fields(KEY, recs))          # spec5 §1: "1.1" and bindKeyCheck exactly when an entry carries parts
    return [head, file_rec, *by_type["configEntry"], *by_type["launchFact"]]


@pytest.mark.parametrize("path", _GOLDEN_PATHS)
def test_each_files_records_validate_as_a_bundle(repo1, path):
    try:
        bundle.validate(_bundle_for(repo1, path))
    except bundle.BundleError as e:                      # a harness fault (header shape) must not hide a record fault
        pytest.fail(str(e))


def test_validator_would_reject_a_bind_value_the_writer_withheld():
    """Positive control for the previous test: the validator is really judging the records."""
    r = run({"a.json": '{"port": "8200"}'})
    recs = _bundle_for(r, "a.json")
    next(x for x in recs if x["type"] == "configEntry")["bindValue"] = "x7k2q9w4m1p8z3v6a1b2c3d4"
    with pytest.raises(bundle.BundleError):
        bundle.validate(recs)


def test_example_files_carry_the_example_role_and_flag(repo1):
    assert repo1.roles["examples/compose.example.yaml"] == repo1.roles[".env.example"] == "example"
    assert repo1.roles["compose.yaml"] == "primary"
    assert all(r["example"] for r in repo1.records if r["type"] == "configEntry" and r["path"] in (".env.example", "examples/compose.example.yaml"))
    assert not any(r["example"] for r in repo1.records if r["type"] == "configEntry" and r["path"] == "compose.yaml")


@pytest.mark.parametrize("path", ["a.example.json", "x.sample.yaml", "example.toml", "sample.ini", "examples/a.json", "a/samples/b/c.yaml",
                                  "Dockerfile.example", ".env.sample", "compose.yaml", "src/examplesx/a.json", "a/b/exampl.json",
                                  "deploy/Compose.EXAMPLE.yml"])
def test_file_role_matches_the_normalizers_rule(path):
    assert X.file_role(path) == assemble_bundle.file_role(path)


def test_file_role_table():
    assert [X.file_role(p) for p in ("a.example.json", "examples/x.yaml", "compose.yaml", "sample.env")] == [
        "example", "example", "primary", "example"]


# --- alias / merge-key (W3) --------------------------------------------------------------------------------------

def test_w3_alias_binds_only_if_both_ends_pass_the_name_rule(repo1):
    b = binds(repo1, "app.yaml")
    assert b["svc.port"] == "8100" and b["svc.peer_port"] == "8100"                 # anchor `svc.port` / alias `svc.peer_port`
    assert b["svc.url"] == "http://localhost:9100/" and "svc.secret_url" not in b   # a merged entry judged at both ends
    assert not any(k.startswith("svc.alias_secret_url") for k in b)                 # alias key fails the name rule


def test_w3_an_anchor_whose_key_fails_the_name_rule_withholds_the_alias():
    r = run({"a.yaml": "secret_port: &sp 8100\nport: *sp\nplain_port: 8100\n"})
    assert binds(r, "a.yaml") == {"plain_port": "8100"}                              # `secret_port` fails; `port` (alias of it) too


def test_w3_a_scalar_alias_with_a_clean_anchor_binds():
    r = run({"a.yaml": "x: &a 8100\nport: *a\n"})
    assert binds(r, "a.yaml") == {"port": "8100"}


def test_w3_a_merged_entry_is_judged_by_the_anchors_key_not_the_merging_maps():
    r = run({"a.yaml": "secret_cfg: &sc {url: 'http://localhost:1/'}\nclean_cfg: &cc {url: 'http://localhost:2/'}\n"
                       "svc:\n  <<: [*sc, *cc]\n"})
    assert binds(r, "a.yaml") == {"clean_cfg.url": "http://localhost:2/", "svc.url": "http://localhost:2/"}
    r2 = run({"a.yaml": "secret_cfg: &sc {url: 'http://localhost:1/'}\nsvc:\n  <<: *sc\n"})
    assert binds(r2, "a.yaml") == {}                                    # `secret_cfg.url` fails; `svc.url` alone would pass


def test_w3_an_inline_merge_has_no_known_anchor_and_withholds():
    r = run({"a.yaml": "m:\n  <<: {url: 'http://localhost:1/'}\n  x_url: 'http://localhost:2/'\n"})
    assert binds(r, "a.yaml") == {"m.x_url": "http://localhost:2/"}


def test_yaml_merge_explicit_keys_win_and_merged_come_first():
    r = run({"a.yaml": "b: &b {url: 'http://localhost:1/', other: 1}\nm:\n  <<: *b\n  url: 'http://localhost:2/'\n"})
    assert entries(r, "a.yaml")["m.url"]["bindValue"] == "http://localhost:2/"


# --- hostile input ---------------------------------------------------------------------------------------------

def test_a_non_utf8_file_refuses_the_whole_unit_naming_the_file_only():
    with pytest.raises(SourceRefused, match="bad.json") as e:
        run({"ok.json": "{}", "bad.json": b'{"k": "\xff\xfe secret-value"}'})
    assert "secret-value" not in str(e.value)


def test_a_utf8_bom_is_accepted():
    assert entries(run({"a.json": b'\xef\xbb\xbf{"port": 8}'}), "a.json")["port"]["bindValue"] == "8"


@pytest.mark.parametrize("path", ["../a.json", "/abs.json", "a//b.json", "a/../b.json", "", "a\\b.json", "./a.json"])
def test_hostile_paths_refuse(path):
    with pytest.raises(SourceRefused):
        run({path: "{}"})


def test_text_with_no_utf8_encoding_drops_the_whole_entry_and_is_counted():
    r = run({"a.yaml": 'good_port: 8000\nbad: "\\uD800"\n"k\\uD800": v\nother: 1\n',
             "b.json": '{"x": "\\ud800", "y_port": "9"}'})
    assert set(entries(r, "a.yaml")) == {"good_port", "other"} and set(entries(r, "b.json")) == {"y_port"}
    assert r.report["unencodableEntries"] == 3
    wr.audit_records(r.records, key=KEY, declared_services=r.declared_services)


def test_an_argv_element_with_no_encoding_drops_the_fact_and_keeps_ordinals_dense():
    r = run({"compose.yaml": 'services:\n  a:\n    command: ["run", "\\uD800"]\n  b:\n    command: ["ok"]\n'})
    f = facts(r, "compose.yaml", "command")
    assert [(x["service"], x["ordinal"]) for x in f] == [("b", 0)] and r.report["unencodableEntries"] >= 1


def test_a_lone_surrogate_default_drops_the_launch_fact():
    r = run({"compose.yaml": 'services:\n  a:\n    environment:\n      A_PORT: "${A_PORT:-\\uD800}"\n'})
    assert facts(r, "compose.yaml", "env-default") == []
    assert set(entries(r, "compose.yaml")) == set() and r.report["unencodableEntries"] >= 1


def _bomb(leaf: str) -> str:
    lines = [f"a0: &a0 [{leaf}]"] + [f"a{i}: &a{i} [{', '.join(f'*a{i - 1}' for _ in range(9))}]" for i in range(1, 12)]
    return "\n".join(lines) + "\n"


@contextlib.contextmanager
def deadline(seconds: float):
    """A test that must bound a hang fails with an error instead of hanging (SIGALRM; main thread only)."""
    def fire(_sig, _frm):
        raise TimeoutError(f"still running after {seconds} s")
    old = signal.signal(signal.SIGALRM, fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


@pytest.mark.parametrize("leaf", ["x, x, x", ""], ids=["leaves", "no-leaves"])
def test_yaml_alias_bomb_is_bounded_structurally(leaf):
    """9**11 expansions. The structural assertion is the walker's own node count; the alarm only turns a regression into a failure."""
    tree = F.parse_yaml(_bomb(leaf))
    w = F.Walk(tree)
    with deadline(20):
        list(w)
    assert w.truncated and w.visited == F.MAX_NODES + 1
    r = run({"bomb.yaml": _bomb(leaf)})
    assert r.report["truncatedFiles"] == 1 and len(entries(r, "bomb.yaml")) <= F.MAX_ENTRIES


def test_a_chain_of_aliases_cannot_nest_deeper_than_the_depth_cap():
    lines = ["a0: &a0 [x]"] + [f"a{i}: &a{i} [*a{i - 1}]" for i in range(1, 45)]
    r = run({"chain.yaml": "\n".join(lines) + "\n"})
    assert r.report["truncatedFiles"] == 1
    assert all(k.count("[") <= F.MAX_DEPTH for k in entries(r, "chain.yaml"))


def test_deep_json_is_skipped_not_recursed():
    r = run({"d.json": "[" * 5000 + "]" * 5000, "ok.json": '{"a": 1}'})
    assert r.report["skipped"] == [{"path": "d.json", "reason": "depth"}] and "a" in entries(r, "ok.json")


def test_deep_yaml_flow_is_skipped():
    r = run({"d.yaml": "[" * 20000 + "]" * 20000 + "\n"})
    assert [s["path"] for s in r.report["skipped"]] == ["d.yaml"] and not entries(r, "d.yaml")


def test_deep_block_yaml_and_deep_toml_are_bounded():
    def yml(n):
        return "".join("  " * i + f"k{i}:\n" for i in range(n)) + "  " * n + "port: 1\n"

    def toml(n):
        return "[" + ".".join(f"t{i}" for i in range(n)) + "]\nport = 1\n"
    r = run({"deep.yaml": yml(60), "deep.toml": toml(60), "ok.yaml": yml(20), "ok.toml": toml(20)})
    assert r.report["skipped"] == [{"path": "deep.toml", "reason": "depth"}, {"path": "deep.yaml", "reason": "depth"}]
    assert len(entries(r, "ok.yaml")) == 1 and len(entries(r, "ok.toml")) == 1


def test_an_oversize_file_is_skipped_and_counted():
    r = run({"big.json": '{"a": "' + "x" * F.MAX_BYTES + '"}', "ok.json": "{}"})
    assert r.report["skipped"] == [{"path": "big.json", "reason": "oversize"}]


def test_an_entry_cap_truncates_and_counts():
    body = ",".join(f'"k{i}": {i}' for i in range(F.MAX_ENTRIES + 50))
    r = run({"many.json": "{" + body + "}"})
    assert len(entries(r, "many.json")) == F.MAX_ENTRIES and r.report["truncatedFiles"] == 1


def test_an_overlong_key_drops_the_entry_without_echoing_it():
    r = run({"a.json": json.dumps({"k" * (F.MAX_KEY_CHARS + 1): "v", "ok": "v"})})
    assert set(entries(r, "a.json")) == {"ok"} and r.report["droppedEntries"] == 1


@pytest.mark.parametrize("path,text", [
    ("a.json", '{"a": NaN}'), ("a.json", '{"a": 1,}'), ("a.json", "{'a': 1}"), ("a.json", '{"a": 1} x'), ("a.json", ""),
    ("a.json", '{"a": "\x01"}'), ("a.json", '// c\n{"a": 1}'), ("a.json", '{"a": 01}'), ("a.json", '{"a" 1}'),
    ("a.yaml", "a: [1, 2"), ("a.yaml", "a: *nope"), ("a.yaml", "\t- a"), ("a.toml", "a = "), ("a.toml", "[a\nb=1"),
])
def test_unparseable_files_are_skipped_with_a_reason_code_and_no_text(path, text):
    r = run({path: text})
    assert [s["path"] for s in r.report["skipped"]] == [path] and not r.records
    assert set(r.report["skipped"][0]) == {"path", "reason"} and len(r.report["skipped"][0]["reason"]) < 20


def test_yaml_tags_are_never_constructed(tmp_path):
    marker = tmp_path / "pwned"
    r = run({"a.yaml": f"x: !!python/object/apply:os.system ['touch {marker}']\ny: !!python/name:os.system\nz_port: !!binary 8000\n"})
    assert not marker.exists() and "x[0]" in entries(r, "a.yaml") and entries(r, "a.yaml")["x[0]"]["format"] == "yaml"


def test_only_the_first_yaml_document_is_read():
    r = run({"a.yaml": "a: 1\n---\nb: 2\n"})
    assert set(entries(r, "a.yaml")) == {"a"}


def test_duplicate_keys_keep_the_last_value_once():
    r = run({"a.json": '{"p_port": "1", "p_port": "2"}', "b.yaml": "p_port: 1\np_port: 2\n", ".env": "P_PORT=1\nP_PORT=2\n"})
    assert binds(r, "a.json") == {"p_port": "2"} and binds(r, "b.yaml") == {"p_port": "2"} and binds(r, ".env") == {"P_PORT": "2"}
    assert len({x["id"] for x in r.records}) == len(r.records)


def test_a_segment_with_a_dot_or_bracket_is_quoted_and_still_valid():
    r = run({"a.json": json.dumps({"a.b": {"c[0]": 1, "": 2, "n\nl": 3, 'q"q': 4}})})
    ks = sorted(entries(r, "a.json"))
    assert ks == sorted(['["a.b"]["c[0]"]', '["a.b"][""]', '["a.b"]["n\\nl"]', '["a.b"].q"q'])
    bundle.validate(_bundle_for(r, "a.json"))


def test_render_key_path_forms():
    assert F.render_key_path(("a", 0, "b.c", "d")) == 'a[0]["b.c"].d'
    assert F.render_key_path((0, "a")) == "[0].a" and F.render_key_path(("a", "b")) == "a.b"


def test_report_holds_counts_and_paths_never_values(repo1):
    assert set(repo1.report) == {"files", "ignoredFiles", "skippedFiles", "configEntries", "launchFacts", "unencodableEntries",
                                 "droppedEntries", "truncatedFiles", "memoryErrors", "skipped"}
    assert repo1.report["configEntries"] == sum(r["type"] == "configEntry" for r in repo1.records)
    assert repo1.report["launchFacts"] == sum(r["type"] == "launchFact" for r in repo1.records)


def test_unknown_files_are_ignored_and_counted():
    r = run({"a.py": "x = 1", "README.md": "# t"})
    assert r.report["ignoredFiles"] == 2 and not r.records and not r.roles


# --- format specifics ------------------------------------------------------------------------------------------

def test_toml_lines_and_array_of_tables():
    t = ('a = 1\n[x]\nb = """\nline [y]\n"""\nc = [\n  1,\n  2,\n]\nd = {e = 1}\n[[w]]\nn = 1\n[[w]]\nn = 2\n[w.sub]\nz = 3\n'
         '[[w]]\nn = 3\n[w.sub]\nz = 4\n')
    r = run({"a.toml": t})
    e = {k: v["line"] for k, v in entries(r, "a.toml").items()}
    assert e["a"] == 1 and e["x.b"] == 3 and e["x.c[0]"] == e["x.c[1]"] == 6 and e["x.d.e"] == 10
    assert e["w[0].n"] == 12 and e["w[1].n"] == 14 and e["w[1].sub.z"] == 16 and e["w[2].n"] == 18 and e["w[2].sub.z"] == 20


def test_toml_scalars_are_canonical_text():
    r = run({"a.toml": 'i = 1_000\nf = 1.50\nb = true\nd = 2020-01-02\ns = "x"\n'})
    got = {k: v for k, v in entries(r, "a.toml").items()}
    texts = {k: _text_of(r, "a.toml", k) for k in got}
    assert texts == {"i": "1000", "f": "1.50", "b": "true", "d": "2020-01-02", "s": "x"}


def _text_of(res, path, kpath):
    rid = ids.config_entry_id(REPO, path, kpath)
    cand = {"1000", "1.50", "true", "2020-01-02", "x", "false"}
    return next(t for t in cand if next(r for r in res.records if r["id"] == rid)["valueHash"] == _independent_hash(rid, t))


def test_ini_sections_comments_and_continuations():
    r = run({"a.ini": "top = 1\n[s]\nk = v\n  more: continued\n; c\n# c\nn: 2\n[s]\nm = 3\n[a.b]\nx = 1\n"})
    assert set(entries(r, "a.ini")) == {"top", "s.k", "s.n", "s.m", '["a.b"].x'}


def test_env_quotes_export_and_inline_comments():
    r = run({".env": "# c\nexport A_PORT=8200 # c\nB_HOST='localhost'\nC_HOST=\"a\\\"b\"\nD=\n1BAD=x\n"})
    e = entries(r, ".env")
    assert set(e) == {"A_PORT", "B_HOST", "C_HOST", "D"}
    assert e["C_HOST"]["valueHash"] == _independent_hash(e["C_HOST"]["id"], 'a"b')
    assert binds(r, ".env") == {"A_PORT": "8200", "B_HOST": "localhost"}


def test_empty_and_scalar_roots_yield_nothing():
    r = run({"a.json": "3", "b.yaml": "", "c.json": "{}", "d.toml": ""})
    assert not r.records and r.report["skipped"] == []


# --- the guard ------------------------------------------------------------------------------------------------

def test_the_repo_guards_cover_this_extractor():
    names = {p.name for p in (CPG / "extractors" / "config").glob("*.py")}
    assert {"extract.py", "formats.py", "launch.py"} <= names
    import test_writer_rules as g          # the P1-07 guard, applied to every extractor module incl. these
    for p in (CPG / "extractors" / "config").glob("*.py"):
        src = p.read_text(encoding="utf-8")
        assert not g.owned_field_writes(src), p.name
        assert not g.forbidden_references(src), p.name


def test_no_extractor_module_imports_a_key_accessor_or_hashes_by_itself():
    for p in (CPG / "extractors" / "config").glob("*.py"):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
            a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not {"hmac", "hashlib", "ccg.valuehash", "ccg.atrest"} & mods, p.name


def test_the_module_loads_only_safe_yaml_and_never_constructs():
    src = (CPG / "extractors" / "config" / "formats.py").read_text(encoding="utf-8")
    assert "yaml.load(" not in src and "unsafe" not in src.lower() and "full_load" not in src and "eval(" not in src
    assert "SafeLoader" in src


# --- B2: merge-key bombs, MemoryError ----------------------------------------------------------------------------

def _merge_chain(levels: int) -> str:
    lines = ["m0: &m0 {a: 1, b: 2}"] + [f"m{i}: &m{i} {{<<: [*m{i - 1}, *m{i - 1}], k{i}: 1}}" for i in range(1, levels + 1)]
    return "\n".join(lines) + "\n"


def test_a_doubling_merge_chain_is_refused_as_nodes_quickly():
    t0 = time.monotonic()
    r = run({"chain.yaml": _merge_chain(22), "ok.yaml": "a: 1\n"})
    assert r.report["skipped"] == [{"path": "chain.yaml", "reason": "nodes"}] and time.monotonic() - t0 < 1
    assert set(entries(r, "ok.yaml")) == {"a"}


def test_the_merge_chain_under_a_one_gigabyte_address_space_cap():
    """The plan's test: `ulimit -v 1000000` and under a second, run in a child so the cap does not bind the test process."""
    code = ("import resource, sys, time, json\n"
            "resource.setrlimit(resource.RLIMIT_AS, (1000000 * 1024, 1000000 * 1024))\n"
            "from extractors.config import extract as X\n"
            "lines = ['m0: &m0 {a: 1}'] + [f'm{i}: &m{i} {{<<: [*m{i-1}, *m{i-1}]}}' for i in range(1, 23)]\n"
            "t = time.monotonic()\n"
            "r = X.extract_config('r', {'c.yaml': ('\\n'.join(lines) + '\\n').encode()}, key=bytes(32))\n"
            "print(json.dumps([r.report['skipped'], time.monotonic() - t]))\n")
    out = subprocess.run([sys.executable, "-I", "-c", code], cwd=CPG, env={"PYTHONPATH": str(CPG)}, capture_output=True, text=True,
                         timeout=40)
    assert out.returncode == 0, out.stderr[-300:]
    skipped, secs = json.loads(out.stdout)
    assert skipped == [{"path": "c.yaml", "reason": "nodes"}] and secs < 1


def test_a_merge_is_lazy_and_correct_for_a_small_chain():
    r = run({"a.yaml": _merge_chain(3)})
    keys = set(entries(r, "a.yaml"))
    assert {"m3.a", "m3.b", "m3.k3", "m3.k2", "m3.k1"} <= keys


def test_a_flat_node_flood_is_refused_as_nodes():
    r = run({"flood.yaml": "[" + ",".join(["x"] * (F.MAX_NODES + 10)) + "]\n"})
    assert r.report["skipped"] == [{"path": "flood.yaml", "reason": "nodes"}]


def test_a_memory_error_skips_that_file_only_and_is_counted(monkeypatch):
    real = F.FORMATS
    boom = F.FormatRow("json", real[5].matches, lambda text: (_ for _ in ()).throw(MemoryError()), "x" * 30)
    monkeypatch.setattr(F, "FORMATS", tuple(boom if r.name == "json" else r for r in real))
    r = run({"a.json": '{"p_port": 1}', "b.yaml": "p_port: 2\n"})
    assert r.report["skipped"] == [{"path": "a.json", "reason": "memory"}] and r.report["memoryErrors"] == 1
    assert set(entries(r, "b.yaml")) == {"p_port"}


def test_a_memory_error_while_building_records_drops_that_files_records(monkeypatch):
    real = X.L.env_defaults
    monkeypatch.setattr(X.L, "env_defaults", lambda text, line: (_ for _ in ()).throw(MemoryError()) if "BOOM" in text else real(text, line))
    r = run({"a.json": '{"x": "BOOM"}', "b.json": '{"y_port": 2}'})
    assert not [x for x in r.records if x["path"] == "a.json"] and "a.json" not in r.roles
    assert r.report["memoryErrors"] == 1 and set(entries(r, "b.json")) == {"y_port"}


# --- M1: keyPath is an identity field (NFC, stripped) ---------------------------------------------------------

@pytest.mark.parametrize("key", ["cafe\u0301", " lead", "trail ", "\t both\n", "a\u00a0"])
def test_non_nfc_or_untrimmed_keys_still_give_a_valid_bundle(key):
    r = run({"a.json": json.dumps({key: "v", "ok": "v"})})
    bundle.validate(_bundle_for(r, "a.json"))
    want = unicodedata.normalize("NFC", key).strip()
    assert want in entries(r, "a.json") and all(unicodedata.normalize("NFC", k) == k for k in entries(r, "a.json"))


def test_a_key_that_strips_to_empty_is_quoted_and_valid():
    r = run({"a.json": json.dumps({"  ": "v"})})
    assert set(entries(r, "a.json")) == {'[""]'}
    bundle.validate(_bundle_for(r, "a.json"))


def test_nfd_and_nfc_spellings_of_one_key_are_one_entry():
    r = run({"a.json": '{"caf\u00e9": "1", "cafe\u0301": "2"}'})
    assert list(entries(r, "a.json")) == ["caf\u00e9"]


def test_anchor_key_paths_are_normalised_too():
    r = run({"a.yaml": "' x_port ': &p 8100\nport: *p\n"})
    bundle.validate(_bundle_for(r, "a.yaml"))
    assert binds(r, "a.yaml") == {"x_port": "8100", "port": "8100"}


# --- M2: only primary-role Compose files declare services --------------------------------------------------

def test_an_example_compose_file_declares_no_services():
    ex = "services:\n  api:\n    container_name: metadata\n    image: x\n"
    app = "services:\n  web:\n    environment:\n      API_HOST: metadata\n      PEER_HOST: api\n"
    r = run({"docker-compose.example.yml": ex, "compose.yaml": app})
    assert r.declared_services == frozenset({"web"})
    assert binds(r, "compose.yaml") == {}
    both = run({"docker-compose.yml": ex, "compose.yaml": app})
    assert both.declared_services == frozenset({"web", "api", "metadata"})
    assert binds(both, "compose.yaml") == {"services.web.environment.API_HOST": "metadata", "services.web.environment.PEER_HOST": "api"}


def test_a_compose_file_under_examples_declares_nothing_and_an_example_default_is_withheld():
    r = run({"examples/compose.yaml": "services:\n  acme:\n    environment:\n      A_HOST: ${A_HOST:-acme}\n"})
    assert r.declared_services == frozenset()
    assert "default" not in facts(r, "examples/compose.yaml", "env-default")[0]


# --- mount facts -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("ctx", ["/var/run", "/etc", "/", "/home/u/app"])
def test_an_absolute_build_context_is_withheld(ctx):
    r = run({"compose.yaml": f"services:\n  a:\n    build:\n      context: {ctx}\n  b:\n    build: {ctx}\n"})
    assert facts(r, "compose.yaml", "mount") == []


def test_a_relative_build_context_may_leave_the_repo_and_is_kept_as_written():
    r = run({"compose.yaml": "services:\n  a:\n    build: ../../sib\n  b:\n    build: ./x/../y\n"})
    assert [f["mountPath"] for f in facts(r, "compose.yaml", "mount")] == ["../../sib", "./x/../y"]


@pytest.mark.parametrize("ctx", ["../ghp_aB3dE5gH7jK9mN1pQ3sT5vX7zA9cE1gI3", "./x7k2q9w4m1p8z3v6a1b2c3d4", "sk-abcdefghij0123456789xyz/svc"])
def test_a_build_context_segment_that_fails_the_token_screen_is_withheld(ctx):
    assert bundle.screen_fails(ctx.split("/")[1] if ctx.startswith(("../", "./")) else ctx.split("/")[0])    # the premise
    r = run({"compose.yaml": f"services:\n  a:\n    build: {ctx}\n"})
    assert facts(r, "compose.yaml", "mount") == []
    ok = run({"compose.yaml": "services:\n  a:\n    build: ../release-2024-notes\n"})
    assert [f["mountPath"] for f in facts(ok, "compose.yaml", "mount")] == ["../release-2024-notes"]


# --- B1: an args-only argv never reaches the gate as written; link-local hosts never bind ----------------------

def _no_secret_no_flag_command(res, *secrets):
    blob = json.dumps(res.records)
    for sec in secrets:
        assert sec not in blob, sec
    for f in (r for r in res.records if r["type"] == "launchFact" and r["kind"] == "command"):
        assert not f["command"].startswith("-"), f


@pytest.mark.parametrize("argv", [["--password", "hunter2lower"], ["-p", "hunter2lower"], ["--api-key", "HUNTER2UPPER"],
                                  ["--token", "abc"], ["-X", "--password", "pw1"]])
def test_a_flag_first_compose_list_leaks_no_secret_and_command_is_never_a_flag(argv):
    for key in ("command", "entrypoint"):
        r = run({"compose.yaml": f"services:\n  a:\n    {key}: {json.dumps(argv)}\n"})
        f = facts(r, "compose.yaml", "command")
        assert len(f) == 1 and f[0]["command"] == bundle.REDACTED and len(f[0]["args"]) == len(argv)
        _no_secret_no_flag_command(r, "hunter2lower", "HUNTER2UPPER", "pw1")


def test_a_flag_first_compose_string_form_leaks_no_secret():
    r = run({"compose.yaml": "services:\n  a:\n    command: --password hunter2lower\n"})
    f = facts(r, "compose.yaml", "command")
    assert f[0]["command"] == bundle.REDACTED
    _no_secret_no_flag_command(r, "hunter2lower")


def test_a_flag_first_mcp_json_command_leaks_no_secret():
    r = run({".mcp.json": json.dumps({"mcpServers": {"s": {"command": "--password", "args": ["hunter2lower"]},
                                                      "t": {"command": "-p", "args": ["hunter2lower"]},
                                                      "u": {"command": "--api-key", "args": ["HUNTER2UPPER"]}}})})
    assert len(facts(r, ".mcp.json")) == 3 and all(f["command"] == bundle.REDACTED for f in facts(r, ".mcp.json"))
    _no_secret_no_flag_command(r, "hunter2lower", "HUNTER2UPPER")


@pytest.mark.parametrize("cmd", ['CMD ["--password", "hunter2lower"]', "CMD --password hunter2lower", 'CMD ["-p", "hunter2lower"]',
                                 'CMD ["--api-key", "HUNTER2UPPER"]', 'CMD ["run", "--password", "hunter2lower"]',
                                 "CMD run --api-key HUNTER2UPPER"])
def test_a_dockerfile_cmd_beside_an_entrypoint_has_no_program_and_leaks_no_secret(cmd):
    r = run({"Dockerfile": f'FROM x\nENTRYPOINT ["tini", "--"]\n{cmd}\n'})
    f = facts(r, "Dockerfile", "command")
    assert [x["command"] for x in f] == ["tini", bundle.REDACTED]
    _no_secret_no_flag_command(r, "hunter2lower", "HUNTER2UPPER")


def test_a_dockerfile_cmd_alone_keeps_its_program_and_a_flag_first_one_is_detected():
    r = run({"Dockerfile": 'FROM x\nCMD ["run", "--port", "8000"]\n'})
    assert facts(r, "Dockerfile", "command")[0]["command"] == "run"
    r2 = run({"Dockerfile": 'FROM x\nCMD ["--password", "hunter2lower"]\n'})
    assert facts(r2, "Dockerfile", "command")[0]["command"] == bundle.REDACTED
    _no_secret_no_flag_command(r2, "hunter2lower")


def test_compose_command_beside_entrypoint_is_arguments_and_alone_is_a_program():
    both = run({"compose.yaml": "services:\n  a:\n    entrypoint: [tini]\n    command: [run, --password, hunter2lower]\n"})
    assert [f["command"] for f in facts(both, "compose.yaml", "command")] == ["tini", bundle.REDACTED]
    _no_secret_no_flag_command(both, "hunter2lower")
    alone = run({"compose.yaml": "services:\n  a:\n    command: [run, --port, '8000']\n"})
    assert facts(alone, "compose.yaml", "command")[0]["command"] == "run"


def test_program_known_is_passed_from_the_drafts(monkeypatch):
    seen = []
    real = wr.launch_argv_fields

    def spy(argv, **kw):
        seen.append((argv[0], kw.get("program_known")))
        return real(argv, **kw)
    monkeypatch.setattr(X.wr, "launch_argv_fields", spy)
    run({"Dockerfile": 'FROM x\nENTRYPOINT ["tini"]\nCMD ["run"]\n'})
    assert seen == [("tini", True), ("run", False)]


@pytest.mark.parametrize("value", ["169.254.169.254", "http://169.254.169.254/latest", "http://169.254.169.254:80/", "169.254.0.1:80",
                                   "fe80::1", "[fe80::1]", "http://[fe80::1]:8000/", "fd00:ec2::254", "[fd00:ec2::254]",
                                   "http://[fd00:ec2::254]/latest", "100.100.100.200", "http://100.100.100.200/"])
def test_link_local_and_metadata_hosts_never_bind(value):
    r = run({"a.json": json.dumps({"api_url": value, "api_host": value, "svc_addr": value, "ok_url": "http://localhost:8000/"})})
    assert binds(r, "a.json") == {"ok_url": "http://localhost:8000/"}
    assert value not in json.dumps(facts(r, "a.json"))


def test_a_link_local_env_default_is_withheld_and_hashed():
    r = run({"compose.yaml": "services:\n  a:\n    environment:\n      A_HOST: ${A_HOST:-169.254.169.254}\n      B_HOST: ${B_HOST:-10.0.0.5}\n"})
    f = {x["envVar"]: x for x in facts(r, "compose.yaml", "env-default")}
    assert "default" not in f["A_HOST"] and "defaultHash" in f["A_HOST"] and f["B_HOST"]["default"] == "10.0.0.5"


# --- N1: service names are identity strings (NFC, stripped), in `service` and in the declared set ---------------

NFD = "cafe\u0301"
NFC = "caf\u00e9"


def test_an_nfd_mcp_server_name_gives_a_valid_bundle_with_the_nfc_service():
    r = run({".mcp.json": json.dumps({"mcpServers": {NFD: {"command": "run"}, " pad ": {"command": "run2"}}})})
    assert sorted(f["service"] for f in facts(r, ".mcp.json")) == sorted([NFC, "pad"])
    bundle.validate(_bundle_for(r, ".mcp.json"))


def test_an_nfd_compose_service_gives_a_valid_bundle_with_the_nfc_service():
    r = run({"compose.yaml": f"services:\n  {NFD}:\n    command: [run]\n    ports: ['8000']\n    build: ./x\n"})
    f = facts(r, "compose.yaml")
    assert f and {x["service"] for x in f} == {NFC}
    bundle.validate(_bundle_for(r, "compose.yaml"))
    assert NFC in r.declared_services and NFD not in r.declared_services


def test_the_declared_set_uses_the_same_normalised_names_so_a_bind_value_still_resolves():
    app = "services:\n  ' api ':\n    image: x\n  web:\n    environment:\n      API_HOST: api\n"
    r = run({"compose.yaml": app})
    assert {"api", "web"} <= r.declared_services and binds(r, "compose.yaml") == {"services.web.environment.API_HOST": "api"}
    assert " api " not in r.declared_services
    r3 = run({"compose.yaml": "services:\n  a:\n    container_name: ' box '\n    environment:\n      B_HOST: box\n"})
    assert "box" in r3.declared_services and binds(r3, "compose.yaml") == {"services.a.environment.B_HOST": "box"}
    r2 = run({"web.compose.yaml": "services:\n  web:\n    environment:\n      A_HOST: peer\n"}, extra_declared_services=[" peer "])
    assert binds(r2, "web.compose.yaml") == {"services.web.environment.A_HOST": "peer"}


# --- P1-14: bindParts (spec5 §6.1, §6.2, §1) --------------------------------------------------------------------

def _parts(res: X.ConfigResult, path: str) -> dict[str, dict]:
    return {k: r["bindParts"] for k, r in entries(res, path).items() if "bindParts" in r}


def test_url_entries_carry_parts_and_the_capability_url_carries_no_bind_value():
    r = run({"a.json": json.dumps({"notify": {"url": "https://ntfy.sh/x7k2-q9w4-m1p8-z3v6"}, "svc_url": "http://localhost:8000/mcp",
                                   "base_url": "http://localhost:8000/api/v1", "port": "8200", "timeout_url": "http://localhost:1/"})})
    e = entries(r, "a.json")
    assert "bindValue" not in e["notify.url"] and e["notify.url"]["bindParts"]["hostKind"] == "external"
    assert "host" not in e["notify.url"]["bindParts"]
    assert e["svc_url"]["bindValue"] == "http://localhost:8000/mcp" and e["svc_url"]["bindParts"]["host"] == "localhost"
    assert "bindValue" not in e["base_url"] and len(e["base_url"]["bindParts"]["pathHmacs"]) == 2    # a multi-segment base binds by parts
    assert "bindParts" not in e["port"] and "bindValue" in e["port"]                               # a non-URL value has none
    assert "bindParts" in e["timeout_url"]                                                          # `url` is a role token here
    for rec in e.values():                                                                          # the URL text itself is never stored
        assert "ntfy.sh" not in json.dumps({k: v for k, v in rec.items() if k != "keyPath"})


def test_parts_are_the_keyed_parts_of_the_entrys_text_under_the_run_key():
    r = run({"compose.yaml": "services:\n  api:\n    environment:\n      DB_URL: redis://api:6379\n      EXT_URL: https://example.org/v1\n"})
    e = entries(r, "compose.yaml")
    for kp, text in (("services.api.environment.DB_URL", "redis://api:6379"), ("services.api.environment.EXT_URL", "https://example.org/v1")):
        assert e[kp]["bindParts"] == valuehash.bind_parts(KEY, text, {"api"})
    assert e["services.api.environment.DB_URL"]["bindParts"]["hostKind"] == "service"
    assert e["services.api.environment.EXT_URL"]["bindParts"]["hostKind"] == "external"
    other = X.extract_config(REPO, {"compose.yaml": "services:\n  api:\n    environment:\n      DB_URL: redis://api:6379\n".encode()},
                             key=bytes(range(96, 128)))
    assert entries(other, "compose.yaml")["services.api.environment.DB_URL"]["bindParts"]["hostHmac"] != \
        e["services.api.environment.DB_URL"]["bindParts"]["hostHmac"]                            # the key matters


def test_every_bind_parts_traces_to_bind_parts_fields(monkeypatch):
    seen: list[str] = []
    real = wr.bind_parts_fields

    def spy(key, kpath, text, **kw):
        out = real(key, kpath, text, **kw)
        if out:
            seen.append(kpath)
        return out
    monkeypatch.setattr(X.wr, "bind_parts_fields", spy)
    res = X.extract_config(REPO, repo_files(), key=KEY)
    emitted = [r["keyPath"] for r in res.records if r["type"] == "configEntry" and "bindParts" in r]
    assert emitted and sorted(emitted) == sorted(seen)
    monkeypatch.setattr(X.wr, "bind_parts_fields", lambda *a, **k: {})
    again = X.extract_config(REPO, repo_files(), key=KEY)
    assert not any("bindParts" in r for r in again.records) and again.header_fields == {}


def test_the_header_fields_follow_the_version_rule():
    with_url = run({"a.json": '{"svc_url": "http://localhost:8000/"}'})
    assert with_url.header_fields == {"bundleSchemaVersion": "1.1", "bindKeyCheck": atrest.bind_key_check(KEY)}
    assert run({"a.json": '{"port": "8200"}'}).header_fields == {}                              # no parts: the header stays "1.0"
    assert run({}).header_fields == {}


@pytest.mark.parametrize("path", _GOLDEN_PATHS)
def test_each_file_header_is_one_point_one_exactly_when_it_has_parts(repo1, path):
    recs = _bundle_for(repo1, path)
    has = any("bindParts" in r for r in recs if r["type"] == "configEntry")
    assert recs[0]["bundleSchemaVersion"] == ("1.1" if has else "1.0") and ("bindKeyCheck" in recs[0]) == has
    bundle.validate(recs)


def test_the_fixture_repo_has_parts_on_url_entries_and_none_on_other_values(repo1):
    texts = _audit_texts(repo1)
    cfg = [r for r in repo1.records if r["type"] == "configEntry"]
    parts = [r for r in cfg if "bindParts" in r]
    assert len(parts) >= 5
    for r in cfg:
        if "bindValue" in r and "://" in r["bindValue"]:   # W7: no clear URL without parts
            assert "bindParts" in r, r["keyPath"]
        if "bindParts" in r:
            assert wr.bind_parts_allowed(r["keyPath"], texts[r["id"]], binding_role=True,
                                         declared_services=repo1.declared_services), r["keyPath"]
            assert bundle.parse_bind_url(texts[r["id"]]) is not None


@pytest.mark.parametrize("value", ["http://169.254.169.254/latest", "http://169.254.169.254:80/", "http://[fe80::1]:8000/",
                                   "http://[fd00:ec2::254]/latest", "http://100.100.100.200/"])
def test_a_link_local_url_gets_external_parts_with_no_clear_host_and_no_bind_value(value):
    r = run({"a.json": json.dumps({"api_url": value})})
    e = entries(r, "a.json")["api_url"]
    host = bundle.parse_bind_url(value).host
    assert "bindValue" not in e and e["bindParts"]["hostKind"] == "external" and "host" not in e["bindParts"]
    assert host not in json.dumps(e) and host.strip("[]") not in json.dumps(e)
    bundle.validate(_bundle_for(r, "a.json"))


def test_an_argv_element_url_and_a_url_without_a_role_carry_no_parts():
    r = run({"compose.yaml": "services:\n  a:\n    command: [\"app\", \"http://localhost:8000/mcp\"]\n    labels:\n      note: http://localhost:8000/x\n"})
    assert not _parts(r, "compose.yaml")


def test_a_lone_surrogate_url_drops_the_entry_as_before_and_never_raises():
    r = run({"a.yaml": 'svc_url: "http://x\\uD800/"\nok_url: "http://localhost:8000/"\n'})
    assert r.report["unencodableEntries"] == 1 and "ok_url" in entries(r, "a.yaml")


def test_a_yaml_alias_gets_parts_only_if_both_ends_pass_the_name_rule():
    clean = run({"a.yaml": 'base_url: &a "http://localhost:9000/"\nsvc_url: *a\n'})
    assert set(_parts(clean, "a.yaml")) == {"base_url", "svc_url"}
    dirty = run({"a.yaml": 'secret_url: &a "http://localhost:9000/"\nsvc_url: *a\n'})        # the anchor's key fails the name rule (W3)
    e = entries(dirty, "a.yaml")
    assert "bindParts" not in e["secret_url"] and "bindParts" not in e["svc_url"] and "bindValue" not in e["svc_url"]


def test_audit_texts_cover_every_entry_that_carries_bind_parts_and_the_audit_finds_nothing(repo1):
    """P1-11 Done: `audit_records` gets the raw text of every `bindParts` entry, so it can recompute the parts."""
    texts = _audit_texts(repo1)
    with_parts = [r["id"] for r in repo1.records if r["type"] == "configEntry" and "bindParts" in r]
    assert with_parts and all(i in texts for i in with_parts)
    assert wr.audit_records(repo1.records, key=KEY, declared_services=repo1.declared_services, texts=texts) == []
    wrong = {**texts, with_parts[0]: "http://elsewhere.example.org/x"}              # the text decides: other text, other parts
    assert wr.audit_records(repo1.records, key=KEY, declared_services=repo1.declared_services, texts=wrong)


def test_audit_over_an_adhoc_run_with_urls_covers_every_parts_entry():
    values = {"notify.url": "https://ntfy.sh/x7k2-q9w4-m1p8-z3v6", "svc_url": "http://localhost:8000/mcp", "a_url": "http://168.63.129.16/x",
              "b_url": "http://[2002:a00:1::1]/", "c_url": "http://10.0.0.5:80/api"}
    r = run({"a.json": json.dumps({"notify": {"url": values["notify.url"]}, **{k: v for k, v in values.items() if "." not in k}})})
    texts = {rec["id"]: values[rec["keyPath"]] for rec in r.records if rec["type"] == "configEntry"}
    parts = [rec for rec in r.records if rec["type"] == "configEntry" and "bindParts" in rec]
    assert len(parts) == 5 and all(rec["id"] in texts for rec in parts)
    assert wr.audit_records(r.records, key=KEY, declared_services=r.declared_services, texts=texts) == []
    for rec in parts:                                                                  # link-local, 6to4, capability: never a clear host
        if rec["keyPath"] in ("a_url", "b_url", "notify.url"):
            assert rec["bindParts"]["hostKind"] == "external" and "host" not in rec["bindParts"]


# --- P1-11b: the `.mcp.json` exec-path launch fact (the allow-listed `bash -c` exception) ------------------------

_PD = "$CLAUDE_PROJECT_DIR"


def _mcp_facts(command, args, name="cypher"):
    r = run({".mcp.json": json.dumps({"mcpServers": {name: {"command": command, "args": args}}})})
    assert wr.audit_records(r.records, key=KEY, declared_services=frozenset(), texts=None) == []
    bundle.validate(_bundle_for(r, ".mcp.json"))
    return facts(r, ".mcp.json", "command")


def test_p1_11b_the_repos_own_mcp_json_names_its_script():
    real = (CPG.parent / ".mcp.json").read_bytes()                         # read-only golden: this repo's own file
    r = X.extract_config(REPO, {".mcp.json": real}, key=KEY)
    f = facts(r, ".mcp.json", "command")
    cypher = [x for x in f if x["service"] == "cypher"]
    assert len(cypher) == 1 and (cypher[0]["command"], cypher[0]["args"]) == ("cypher-mcp/docker-run.sh", [])
    assert not any(a == "-c" or bundle.REDACTED in a for x in f for a in x["args"])
    assert wr.audit_records(r.records, key=KEY, declared_services=frozenset()) == []


@pytest.mark.parametrize("shell", ["bash", "sh"])
def test_p1_11b_an_allow_listed_launch_emits_the_bare_path_with_its_service_and_line(shell):
    (f,) = _mcp_facts(shell, ["-c", f'exec "{_PD}/scripts/run-mcp.sh"'])
    assert (f["command"], f["args"], f["service"], f["origin"], f["kind"]) == ("scripts/run-mcp.sh", [], "cypher", "explicit", "command")


@pytest.mark.parametrize("s", [
    f'exec x; curl y', f'exec "{_PD}/x"; curl y', f'exec "{_PD}/$(id)"', f'exec "{_PD}/`id`"', f'exec "{_PD}/../x"',
    f'exec "{_PD}/ghp_abcdefghijklmnopqrstuvwxyz0123456789/run"', "exec /etc/passwd", f'exec "{_PD}/x" extra',
    f"exec '{_PD}/x'", f'exec "{_PD}/x"\n', f'exec "{_PD}/x" && curl y', f'exec "{_PD}/x" | sh', f'exec "{_PD}/x" > f',
    f'FOO=bar exec "{_PD}/x"', f'exec "{_PD}/\u0430/x"', f'exec "{_PD}/x\x00"', f'exec "{_PD}/x "', "exec " + "a" * 4000,
    f'exec "{_PD}//etc/passwd"',
])
@pytest.mark.parametrize("shell", ["bash", "sh"])
def test_p1_11b_every_other_bash_c_string_stays_redacted(shell, s):
    (f,) = _mcp_facts(shell, ["-c", s])
    assert f["command"] == shell and f["args"][0] == "-c" and len(f["args"]) == 2          # the gate's own result, never the bare path
    assert f["args"] == bundle.gate_argv([shell, "-c", s])[1:]


@pytest.mark.parametrize("s", ["exec x; curl y", f'exec "{_PD}/$(id)"', f'exec "{_PD}/../x"', f'exec "{_PD}/`id`"'])
def test_p1_11b_the_headline_negatives_are_redacted_not_merely_gated(s):
    (f,) = _mcp_facts("bash", ["-c", s])
    assert (f["command"], f["args"]) == ("bash", ["-c", bundle.REDACTED])


@pytest.mark.parametrize("args", [["-c", f'exec "{_PD}/x"', "extra"], ["-lc", f'exec "{_PD}/x"'], ["-e", "-c", f'exec "{_PD}/x"']])
def test_p1_11b_extra_or_different_args_stay_gated(args):
    (f,) = _mcp_facts("bash", args)
    assert f["command"] == "bash" and f["args"] and f["args"] != []
    assert not any("CLAUDE_PROJECT_DIR" in a for a in f["args"])


def test_p1_11b_the_exception_is_mcp_json_only():
    s = f'exec "{_PD}/x/run.sh"'
    r = run({"compose.yaml": f"services:\n  a:\n    command: [\"bash\", \"-c\", {json.dumps(s)}]\n",
             "Dockerfile": f"FROM x\nENTRYPOINT [\"bash\", \"-c\", {json.dumps(s)}]\n"})
    for p in ("compose.yaml", "Dockerfile"):
        (f,) = facts(r, p, "command")
        assert (f["command"], f["args"]) == ("bash", ["-c", bundle.REDACTED])


def test_p1_11b_a_kiro_agent_gets_the_same_rule_as_mcp_json_or_none():
    s = f'exec "{_PD}/x/run.sh"'
    r = run({".kiro/agents/a.json": json.dumps({"mcpServers": {"s": {"command": "bash", "args": ["-c", s]}}})})
    (f,) = facts(r, ".kiro/agents/a.json", "command")
    assert (f["command"], f["args"]) == ("bash", ["-c", bundle.REDACTED])


def test_p1_11b_an_over_long_exec_string_yields_no_fact_at_all():
    assert _mcp_facts("bash", ["-c", "exec " + "a" * 5000]) == []


def test_p1_11b_a_path_over_128_characters_keeps_the_full_redaction_not_a_bare_redacted_command():
    path = ("ab/" * 43)[:128] + "z"                                # 129 characters
    (f,) = _mcp_facts("bash", ["-c", f'exec "{_PD}/{path}"'])
    assert (f["command"], f["args"]) == ("bash", ["-c", bundle.REDACTED])
