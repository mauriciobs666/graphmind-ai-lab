"""P1-03 — the system manifest loader, the egress section's S1-S5 checks, `canonical_endpoint`, the two schemas.

Plan: `docs/plans/connected-code-graph-p1-extraction.md` § P1-03. Egress: `docs/plans/connected-code-graph-egress-spec3.md`
§4 (schema, S1-S5, per-repository mode) and §10.1 (P1-03's tests). Expected values are literals written here; the
two files compared to a source (`egress-v1.schema.json` to the spec's block, `system.yaml`'s exclusions to
`cpg/eval/eval-manifest.yaml`) are compared to an independent artifact on purpose.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest
import yaml

from ccg import manifest as mf
from ccg.manifest import ManifestError

CPG = Path(__file__).resolve().parent.parent
REPO = CPG.parent
SYSTEM_YAML = CPG / "ccg" / "system.yaml"
FIXTURE_YAML = CPG / "fixtures" / "system1" / "system.yaml"
EGRESS_SCHEMA = CPG / "ccg" / "schemas" / "egress-v1.schema.json"
MANIFEST_SCHEMA = CPG / "ccg" / "schemas" / "system-manifest-v1.schema.json"
SPEC3 = REPO / "docs" / "plans" / "connected-code-graph-egress-spec3.md"
EVAL_MANIFEST = CPG / "eval" / "eval-manifest.yaml"

GOOD_EGRESS = {
    "schemaVersion": 1,
    "local": [
        {"id": "lmstudio", "kind": "model", "endpoint": "http://127.0.0.1:1234", "paths": ["/v1/chat/completions"]}
    ],
    "outbound": [],
    "hosted": [],
}


def good() -> dict:
    return {
        "schemaVersion": 1,
        "system": "shop",
        "egress": copy.deepcopy(GOOD_EGRESS),
        "repos": [{"repoId": "Shop Front", "path": ".", "policy": {"egress": "local-only"}}],
    }


def write(tmp_path: Path, doc, name="system.yaml") -> Path:
    p = tmp_path / name
    p.write_text(doc if isinstance(doc, str) else yaml.safe_dump(doc, sort_keys=False))
    return p


def egress_codes(egress: dict) -> tuple[str, ...]:
    with pytest.raises(ManifestError) as e:
        mf.check_egress(egress)
    return tuple(e.value.codes)


# --- the two shipped manifests -----------------------------------------------------------------------------------


def test_the_proving_system_manifest_loads_with_the_fixed_repo_id_and_a_local_only_mode():
    m = mf.load_manifest(SYSTEM_YAML)
    assert m.system == "graphmind-ai-lab"
    assert [r.repo_id for r in m.repos] == ["graphmind-ai-lab"]
    assert m.repos[0].path == "."
    assert m.repos[0].policy.egress == "local-only"
    assert m.egress["local"][0]["endpoint"] == "http://127.0.0.1:1234"


def test_the_fixture_system_manifest_has_its_four_repos_each_with_a_mode():
    m = mf.load_manifest(FIXTURE_YAML)
    assert [(r.repo_id, r.path) for r in m.repos] == [
        ("fx-svc", "fx-svc"), ("fx-web", "fx-web"), ("fx-lib", "fx-lib"), ("fx-job", "fx-job"),
    ]
    assert {r.policy.egress for r in m.repos} == {"local-only"}


@pytest.mark.parametrize("path", [SYSTEM_YAML, FIXTURE_YAML])
def test_policy_egress_is_written_on_every_repository_of_the_shipped_manifests(path):
    raw = yaml.safe_load(path.read_text())
    assert raw["repos"] and all(r["policy"]["egress"] in ("local-only", "hosted") for r in raw["repos"])
    assert "endpointAllowList" not in raw and all("localOnly" not in r["policy"] for r in raw["repos"])


def test_the_proving_systems_exclusions_equal_the_census_scope_of_the_eval_manifest():
    ev = yaml.safe_load(EVAL_MANIFEST.read_text())["exclusions"]
    repo = mf.load_manifest(SYSTEM_YAML).repos[0]
    assert list(repo.exclusion_paths) == ev["paths"]
    assert list(repo.content_check_files) == [f["path"] for f in ev["contentCheck"]["files"]]
    # Anchors so the comparison cannot pass on two empty lists, and the named scope is really there.
    assert "cpg/" in repo.exclusion_paths and "deprecated/" in repo.exclusion_paths
    assert "docs/BACKLOG.md" in repo.exclusion_paths and len(repo.content_check_files) == 16


def test_the_default_preview_cap_is_two_and_the_windows_are_optional(tmp_path):
    r = mf.load_manifest(write(tmp_path, good())).repos[0]
    assert r.policy.preview_cap == 2
    assert r.policy.rca_window is None and r.policy.secrets_window is None


def test_policy_windows_and_cap_are_read(tmp_path):
    d = good()
    d["repos"][0]["policy"].update({"rcaWindow": 14, "secretsWindow": 365, "previewCap": 3})
    p = mf.load_manifest(write(tmp_path, d)).repos[0].policy
    assert (p.rca_window, p.secrets_window, p.preview_cap) == (14, 365, 3)


# --- rejections ----------------------------------------------------------------------------------------------------


def _mutate(fn):
    d = good()
    fn(d)
    return d


REJECTED = {
    "unknown top-level key": lambda d: d.update(extra=1),
    "unknown repo key": lambda d: d["repos"][0].update(extra=1),
    "absent repoId": lambda d: d["repos"][0].pop("repoId"),
    "absent system": lambda d: d.pop("system"),
    "system name with a trailing separator": lambda d: d.update(system="shop_"),
    "system name in capitals": lambda d: d.update(system="Shop"),
    "system name of 49 characters": lambda d: d.update(system="a" * 49),
    "absent policy": lambda d: d["repos"][0].pop("policy"),
    "absent policy.egress": lambda d: d["repos"][0].update(policy={"previewCap": 2}),
    "the boolean twin policy.localOnly": lambda d: d["repos"][0].update(policy={"localOnly": True}),
    "policy.egress outside the enum": lambda d: d["repos"][0]["policy"].update(egress="both"),
    "the removed endpointAllowList": lambda d: d.update(endpointAllowList=["http://x:1"]),
    "a missing egress section": lambda d: d.pop("egress"),
    "schemaVersion 2": lambda d: d.update(schemaVersion=2),
    "no repos": lambda d: d.update(repos=[]),
    "a window of zero": lambda d: d["repos"][0]["policy"].update(rcaWindow=0),
    "a string preview cap": lambda d: d["repos"][0]["policy"].update(previewCap="2"),
    "an unknown exclusions key": lambda d: d["repos"][0].update(exclusions={"globs": []}),
    "an absolute repo path": lambda d: d["repos"][0].update(path="/etc"),
    "a repo path that climbs": lambda d: d["repos"][0].update(path="../x"),
    "an un-normalized repo path": lambda d: d["repos"][0].update(path="a//b"),
    "an exclusion that climbs": lambda d: d["repos"][0].update(exclusions={"paths": ["../x"]}),
    "a promptBearingMarkdown that is not a mapping": lambda d: d.update(promptBearingMarkdown=["a"]),
    "a promptBearingMarkdown without include": lambda d: d.update(promptBearingMarkdown={"except": ["a"]}),
    "an empty include": lambda d: d.update(promptBearingMarkdown={"include": []}),
    "an include that is a string": lambda d: d.update(promptBearingMarkdown={"include": "**/A.md"}),
    "an unknown promptBearingMarkdown key": lambda d: d.update(promptBearingMarkdown={"include": ["a.md"], "also": []}),
    "an absolute glob": lambda d: d.update(promptBearingMarkdown={"include": ["/etc/*.md"]}),
    "a glob that climbs": lambda d: d.update(promptBearingMarkdown={"include": ["a/../b.md"]}),
    "a glob that is a bare ..": lambda d: d.update(promptBearingMarkdown={"include": [".."]}),
    "a glob with a NUL": lambda d: d.update(promptBearingMarkdown={"include": ["a\x00.md"]}),
    "a glob with an empty segment": lambda d: d.update(promptBearingMarkdown={"include": ["a//b.md"]}),
    "a glob with a partial **": lambda d: d.update(promptBearingMarkdown={"include": ["a**b.md"]}),
    "a glob outside the alphabet": lambda d: d.update(promptBearingMarkdown={"include": ["a/[bc].md"]}),
    "a non-string glob": lambda d: d.update(promptBearingMarkdown={"include": [1]}),
    "a 201-character glob": lambda d: d.update(promptBearingMarkdown={"include": ["a" * 201]}),
    "65 include globs": lambda d: d.update(promptBearingMarkdown={"include": [f"d{i}/*.md" for i in range(65)]}),
    "65 except globs": lambda d: d.update(promptBearingMarkdown={"include": ["a.md"], "except": [f"d{i}" for i in range(65)]}),
    "a non-string except glob": lambda d: d.update(promptBearingMarkdown={"include": ["a.md"], "except": [{"a": 1}]}),
    "an empty repoId after normalization": lambda d: d["repos"][0].update(repoId="!!!"),
    "two repos with one normalized name": lambda d: d["repos"].append(
        {"repoId": "shop-front", "path": "b", "policy": {"egress": "local-only"}}
    ),
    "an unknown egress key (registries)": lambda d: d["egress"].update(registries=[]),
    "an unknown egress key (gitFetchMaxBytesOut)": lambda d: d["egress"].update(gitFetchMaxBytesOut=1),
    "an unknown egress key (importNameServices)": lambda d: d["egress"].update(importNameServices=[]),
}


@pytest.mark.parametrize("name", list(REJECTED))
def test_a_planted_manifest_is_refused(tmp_path, name):
    with pytest.raises(ManifestError):
        mf.load_manifest(write(tmp_path, _mutate(REJECTED[name])))


def test_the_good_manifest_loads_so_the_rejections_are_about_their_mutation(tmp_path):
    m = mf.load_manifest(write(tmp_path, good()))
    assert m.system == "shop" and m.repos[0].repo_id == "shop-front"


def test_a_remote_key_is_refused_with_the_p1_message(tmp_path):
    d = good()
    d["repos"][0]["remote"] = "https://example.com/x.git"
    with pytest.raises(ManifestError, match="git fetch is not enabled in P1"):
        mf.load_manifest(write(tmp_path, d))


@pytest.mark.parametrize("remote", [None, "", [], ["https://example.com/x.git"]])
def test_a_remote_key_is_refused_whatever_its_value(tmp_path, remote):
    d = good()
    d["repos"][0]["remote"] = remote
    with pytest.raises(ManifestError, match="git fetch is not enabled in P1"):
        mf.load_manifest(write(tmp_path, d))


def test_groupings_are_refused_until_p2_with_a_clear_error(tmp_path):
    d = good()
    d["groupings"] = [{"name": "g", "repos": ["shop-front"]}]
    with pytest.raises(ManifestError, match=r"groupings.*P2"):
        mf.load_manifest(write(tmp_path, d))


def test_a_system_name_failure_names_the_name_rule(tmp_path):
    d = good()
    d["system"] = "shop_"
    with pytest.raises(ManifestError, match="system name"):
        mf.load_manifest(write(tmp_path, d))


def test_a_duplicate_yaml_key_is_refused_not_silently_resolved(tmp_path):
    text = "schemaVersion: 1\nsystem: a\nsystem: b\negress: {}\nrepos: []\n"
    with pytest.raises(ManifestError, match="duplicate"):
        mf.load_manifest(write(tmp_path, text))


@pytest.mark.parametrize("text", ["- a\n- b\n", "just a string\n", "", "a: [unclosed\n"])
def test_a_manifest_that_is_not_a_yaml_mapping_is_refused(tmp_path, text):
    with pytest.raises(ManifestError):
        mf.load_manifest(write(tmp_path, text))


# --- repoId normalization (bundle spec3 §3.4) ------------------------------------------------------------------


@pytest.mark.parametrize(
    "given, expected",
    [
        ("Online Boutique", "online-boutique"),
        ("graphmind-ai-lab", "graphmind-ai-lab"),
        ("  Foo__Bar  ", "foo__bar"),
        ("a  b!!c", "a-b-c"),
        ("--x--", "x"),
        ("a.b_c-d", "a.b_c-d"),
        ("Café", "caf"),  # NFC first: the accented letter is then outside [a-z0-9._-]
        ("A/B", "a-b"),
    ],
)
def test_repo_ids_are_normalized_per_spec3_section_3_4(given, expected):
    assert mf.normalize_repo_id(given) == expected


@pytest.mark.parametrize("given", ["", "   ", "!!!", "---", "e\u0301"])  # the last: NFC makes it U+00E9, outside the class
def test_a_repo_name_that_normalizes_to_nothing_is_an_error(given):
    with pytest.raises(ManifestError):
        mf.normalize_repo_id(given)


# --- the egress section: S1-S5 --------------------------------------------------------------------------------


def egress_with(**over) -> dict:
    e = copy.deepcopy(GOOD_EGRESS)
    e.update(over)
    return e


def local(id_, endpoint, **kw):
    return {"id": id_, "kind": "model", "endpoint": endpoint, "paths": ["/v1/x"], **kw}


def test_a_valid_egress_section_passes_every_check():
    mf.check_egress(copy.deepcopy(GOOD_EGRESS))
    mf.check_egress(egress_with(
        local=[local("a", "http://127.0.0.1:1234"), local("b", "https://localhost:8443"), local("c", "http://[::1]:80")],
        outbound=[{"id": "osv", "kind": "vulndb", "endpoint": "https://api.osv.dev:443", "paths": ["/v1/query"]}],
        hosted=[{"id": "h1", "kind": "model", "endpoint": "https://api.example.com:443", "paths": ["/v1/m"]}],
    ))


def test_s1_an_id_repeated_across_sections_is_refused():
    e = egress_with(
        local=[local("dup", "http://127.0.0.1:1234")],
        hosted=[{"id": "dup", "kind": "model", "endpoint": "https://api.example.com:443", "paths": ["/v1/m"]}],
    )
    assert egress_codes(e) == ("S1",)


def test_s1_an_id_repeated_within_one_section_is_refused():
    e = egress_with(local=[local("dup", "http://127.0.0.1:1"), local("dup", "http://127.0.0.1:2")])
    assert egress_codes(e) == ("S1",)


def test_s2_an_endpoint_in_two_sections_is_refused():
    e = egress_with(
        local=[local("a", "https://127.0.0.1:8443")],
        outbound=[{"id": "b", "kind": "vulndb", "endpoint": "https://127.0.0.1:8443", "paths": ["/v1/q"]}],
    )
    assert egress_codes(e) == ("S2",)


def test_s2_an_endpoint_in_two_entries_of_one_section_is_refused():
    e = egress_with(local=[local("a", "http://127.0.0.1:1234"), local("b", "http://127.0.0.1:1234")])
    assert egress_codes(e) == ("S2",)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://127.0.0.1:65536",
        "http://127.0.0.1:99999",
        "http://[0:0:0:0:0:0:0:1]:80",  # an IPv6 literal that is not its compressed form
        "http://[0::1]:80",
        "http://[::ffff:127.0.0.1]:80",  # compressed form of a mapped address is ::ffff:7f00:1
    ],
)
def test_s3_a_port_above_65535_or_an_uncompressed_ipv6_literal_is_refused(endpoint):
    assert egress_codes(egress_with(local=[local("a", endpoint)])) == ("S3",)


@pytest.mark.parametrize("port", ["1", "65535", "8000"])
def test_s3_boundary_ports_pass(port):
    mf.check_egress(egress_with(local=[local("a", f"http://127.0.0.1:{port}")]))


@pytest.mark.parametrize(
    "endpoint",
    ["http://0.0.0.0:80", "http://[::]:80", "http://224.0.0.1:80", "http://239.255.255.255:80", "http://[ff02::1]:80"],
)
def test_s4_a_local_unspecified_or_multicast_literal_is_refused(endpoint):
    assert egress_codes(egress_with(local=[local("a", endpoint)])) == ("S4",)


@pytest.mark.parametrize("endpoint", ["http://223.255.255.255:80", "http://240.0.0.1:80", "http://[fe00::1]:80"])
def test_s4_neighbours_of_the_forbidden_ranges_pass_s4(endpoint):
    mf.check_egress(egress_with(local=[local("a", endpoint)]))


def test_s4_applies_to_local_entries_only_not_to_outbound_literals():
    mf.check_egress(egress_with(
        outbound=[{"id": "o", "kind": "vulndb", "endpoint": "https://224.0.0.1:443", "paths": ["/v1/q"]}]
    ))


def test_s5_a_trailing_newline_in_an_endpoint_is_refused_though_the_schema_pattern_admits_it():
    endpoint = "http://127.0.0.1:1234\n"
    # Positive control: the schema alone lets it through (Python's `$` matches before a final newline).
    import jsonschema

    schema = json.loads(EGRESS_SCHEMA.read_text())
    jsonschema.Draft202012Validator(schema).validate(egress_with(local=[local("a", endpoint)]))
    assert egress_codes(egress_with(local=[local("a", endpoint)])) == ("S5",)


@pytest.mark.parametrize(
    "entry",
    [
        local("a\n", "http://127.0.0.1:1234"),  # id: the `id` pattern's `$` admits a final newline too
        local("a", "http://127.0.0.1:1234", note="x\x01y"),  # a control character in a free-text note
        local("a", "http://127.0.0.1:1234", note="x\x7fy"),
        local("a", "http://127.0.0.1:1234", note="x\x1fy"),
    ],
)
def test_s5_a_control_character_anywhere_in_the_section_is_refused(entry):
    assert egress_codes(egress_with(local=[entry])) == ("S5",)


def test_s5_compares_each_endpoint_with_the_one_canonical_endpoint_function(monkeypatch):
    """No schema-valid endpoint that passes S3 differs from its canonical form today, so the comparison's reach is
    shown by substituting the function it must call: S5 follows `canonical_endpoint`, it holds no second rule."""
    mf.check_egress(copy.deepcopy(GOOD_EGRESS))
    monkeypatch.setattr(mf, "canonical_endpoint", lambda e: e + "/")
    assert egress_codes(copy.deepcopy(GOOD_EGRESS)) == ("S5",)


def test_s5_the_boundary_characters_are_admitted():
    mf.check_egress(egress_with(local=[local("a", "http://127.0.0.1:1234", note="space   and ~ and \u0080")]))


def test_every_violated_check_is_reported_not_only_the_first():
    e = egress_with(local=[local("a", "http://0.0.0.0:99999"), local("a", "http://[::]:80")])
    assert set(egress_codes(e)) == {"S1", "S3", "S4"}


@pytest.mark.parametrize(
    "bad",
    [
        {"outbound": [{"id": "o", "kind": "vulndb", "endpoint": "http://api.example.com:80", "paths": ["/v1/q"]}]},
        {"hosted": [{"id": "h", "kind": "model", "endpoint": "http://api.example.com:80", "paths": ["/v1/q"]}]},
        {"local": [local("a", "ftp://127.0.0.1:21")]},
        {"local": [local("a", "http://127.0.0.1")]},  # no port
        {"local": [local("a", "http://example.com:80")]},  # a remote host in `local`
        {"local": [local("A", "http://127.0.0.1:1")]},  # id pattern
        {"local": [local("a", "http://127.0.0.1:1", paths=[])]},
        {"local": [local("a", "http://127.0.0.1:1", paths=["/x", "/x"])]},
        {"local": [local("a", "http://127.0.0.1:1", paths=["x"])]},
        {"local": [local("a", "http://127.0.0.1:1", kind="vulndb")]},
        {"hosted": [{"id": "h", "kind": "vulndb", "endpoint": "https://x.example.com:443", "paths": ["/q"]}]},
        {"local": [local("a", "http://127.0.0.1:1", extra=True)]},
    ],
)
def test_a_schema_violation_is_refused_as_a_manifest_error(bad):
    with pytest.raises(ManifestError):
        mf.check_egress(egress_with(**bad))


def test_a_manifest_with_a_bad_egress_section_is_refused_by_the_loader_with_its_codes(tmp_path):
    d = good()
    d["egress"]["local"] = [local("a", "http://0.0.0.0:80")]
    with pytest.raises(ManifestError) as e:
        mf.load_manifest(write(tmp_path, d))
    assert e.value.codes == ("S4",)


# --- canonical_endpoint ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "given, expected",
    [
        ("http://127.0.0.1:1234", "http://127.0.0.1:1234"),
        ("HTTP://LocalHost:80", "http://localhost:80"),
        ("https://API.Example.COM:443", "https://api.example.com:443"),
        ("http://localhost", "http://localhost:80"),
        ("https://example.com", "https://example.com:443"),
        ("https://example.com:8443/path?q=1#f", "https://example.com:8443"),
        ("https://user:pw@example.com:8443", "https://example.com:8443"),
        ("http://[::1]:80", "http://[::1]:80"),
        ("http://[0:0:0:0:0:0:0:1]:80", "http://[::1]:80"),
        ("http://[2001:DB8:0:0:0:0:0:1]:80", "http://[2001:db8::1]:80"),
        ("http://[2001:db8:0:1:1:1:1:1]:80", "http://[2001:db8:0:1:1:1:1:1]:80"),  # RFC 5952: one zero group stays
    ],
)
def test_canonical_endpoint_forms(given, expected):
    assert mf.canonical_endpoint(given) == expected


@pytest.mark.parametrize(
    "given",
    [
        "", "localhost:80", "ftp://h:21", "http://", "http://h:0", "http://h:65536", "http://h:x",
        "http://h:80\n", "http://h\t:80", "http://h :80", " http://h:80", "http://h:80\x7f", "http://hé:80",
        "http://[::1%eth0]:80", "http://[zz]:80", None, 5,
    ],
)
def test_canonical_endpoint_refuses_the_unparseable(given):
    with pytest.raises(ManifestError):
        mf.canonical_endpoint(given)


def test_canonical_endpoint_is_idempotent_and_a_fixed_point_only_for_the_canonical_form():
    for e in ["http://127.0.0.1:1234", "https://api.example.com:443", "http://[::1]:80"]:
        assert mf.canonical_endpoint(e) == e
        assert mf.canonical_endpoint(mf.canonical_endpoint(e)) == e
    assert mf.canonical_endpoint("http://localhost") != "http://localhost"


# --- the two schemas -------------------------------------------------------------------------------------------


def test_the_egress_schema_is_the_spec_block_byte_for_byte():
    text = SPEC3.read_text()
    section = text[text.index("### 4.2 The schema (normative)"):]
    block = re.search(r"```json\n(.*?)\n```", section, re.S).group(1)
    assert EGRESS_SCHEMA.read_text() == block + "\n"
    assert json.loads(block)["$id"] == "urn:ccg:egress-v1"


def test_the_manifest_schemas_inline_mode_enum_equals_the_egress_schemas_egress_mode():
    egress = json.loads(EGRESS_SCHEMA.read_text())["$defs"]["egressMode"]["enum"]
    ms = json.loads(MANIFEST_SCHEMA.read_text())
    inline = ms["$defs"]["policy"]["properties"]["egress"]["enum"]
    assert inline == egress == ["local-only", "hosted"]


def test_the_manifest_schema_requires_the_mode_and_has_no_boolean_twin():
    ms = json.loads(MANIFEST_SCHEMA.read_text())
    policy = ms["$defs"]["policy"]
    assert "egress" in policy["required"]
    assert "localOnly" not in policy["properties"] and "endpointAllowList" not in ms["properties"]
    assert policy["additionalProperties"] is False and ms["additionalProperties"] is False


# --- P1-03a: promptBearingMarkdown ---------------------------------------------------------------------------------------


def _block(doc: dict) -> dict:
    d = good()
    d["promptBearingMarkdown"] = doc
    return d


def test_the_proving_systems_prompt_globs_equal_the_eval_manifests_block():
    ev = yaml.safe_load(EVAL_MANIFEST.read_text())["promptBearingMarkdown"]
    g = mf.load_manifest(SYSTEM_YAML).prompt_markdown
    assert g is not None
    assert list(g.include) == ev["include"] and list(g.exclude) == ev["except"]
    # anchors so the comparison cannot pass on two empty lists
    assert len(g.include) == 9 and len(g.exclude) == 3 and "claude/**/kaizen/**" in g.exclude


def test_the_fixture_systems_block_classifies_its_prompt_files_prompt_md_and_its_docs_doc():
    g = mf.load_manifest(FIXTURE_YAML).prompt_markdown
    assert g is not None
    for p in (".claude/agents/agent.md", ".claude/agents/tools.md", ".claude/agents/exec.md"):
        assert g.classify(p) == "prompt-md"
    assert g.classify("docs/README.md") == "doc" and g.classify("README.md") == "doc"


def test_a_manifest_without_the_block_loads_with_none(tmp_path):
    assert mf.load_manifest(write(tmp_path, good())).prompt_markdown is None


def test_the_block_is_read_with_include_and_optional_except(tmp_path):
    g = mf.load_manifest(write(tmp_path, _block({"include": ["a/**/*.md"], "except": ["a/x/**"]}))).prompt_markdown
    assert g.include == ("a/**/*.md",) and g.exclude == ("a/x/**",)
    assert g.classify("a/b/c.md") == "prompt-md" and g.classify("a/x/c.md") == "doc"
    g = mf.load_manifest(write(tmp_path, _block({"include": ["a/*.md"]}))).prompt_markdown
    assert g.exclude == ()


def test_the_manifest_and_load_globs_agree_on_the_fixture_prompt_files():
    from extractors.markdown.globs import load_globs
    a, b = mf.load_manifest(FIXTURE_YAML).prompt_markdown, load_globs(FIXTURE_YAML)
    assert (a.include, a.exclude) == (b.include, b.exclude)
    tree = FIXTURE_YAML.parent / "fx-svc"
    mds = sorted(str(p.relative_to(tree)) for p in tree.rglob("*.md"))
    assert len(mds) == 4 and [x for x in mds if b.classify(x) == "prompt-md"] == [x for x in mds if a.classify(x) == "prompt-md"]
    assert [x for x in mds if a.classify(x) == "prompt-md"] == [
        ".claude/agents/agent.md", ".claude/agents/exec.md", ".claude/agents/tools.md"]


@pytest.mark.parametrize("secret", ["SECRET-GLOB-VALUE/../x", "S3CR3T\x00", "S3CR3T[a]", "/S3CR3T"])
def test_a_refused_glob_is_not_echoed_in_the_message(tmp_path, secret):
    with pytest.raises(ManifestError) as e:
        mf.load_manifest(write(tmp_path, _block({"include": [secret]})))
    assert "S3CR3T" not in str(e.value) and "SECRET-GLOB" not in str(e.value) and "promptBearingMarkdown" in str(e.value)


def test_a_duplicate_key_inside_the_block_is_refused(tmp_path):
    text = yaml.safe_dump(good(), sort_keys=False) + "promptBearingMarkdown:\n  include: [a.md]\n  include: [b.md]\n"
    with pytest.raises(ManifestError, match="duplicate"):
        mf.load_manifest(write(tmp_path, text))


def test_the_widest_accepted_block_matches_a_hostile_path_in_bounded_time(tmp_path):
    import time
    inc = ["a*" * 100, "/".join(["**"] * 64)[:200]] + ["**/" * 20 + "*a*a*a*a*b.md"] * 62
    g = mf.load_manifest(write(tmp_path, _block({"include": inc, "except": inc}))).prompt_markdown
    hostile = "/".join(["a" * 255] * 63) + "/" + "a" * 254 + "c.md"
    start = time.monotonic()
    assert g.classify(hostile) == "doc"
    assert time.monotonic() - start < 5


@pytest.mark.parametrize("text", ["? [a]\n: 1\n", "schemaVersion: 1\n? {a: 1}\n: 2\n", "x:\n  ? [a]\n  : 1\n"])
def test_an_unhashable_yaml_key_is_a_typed_value_free_refusal_not_a_type_error(tmp_path, text):
    with pytest.raises(ManifestError) as e:
        mf.load_manifest(write(tmp_path, text))
    assert "scalar" in str(e.value)
