"""§4 S2's real pack loader — `load_pack`, `content_hash`, `validate_pack`, and the §3.3 totality
boundary (plan §4 S2, §5 test 4 and test 12).

Every pack here is a real directory under `tests/fixtures/packs/` (`conftest.pack_fixture`), never
an in-memory manifest: the row-count identity and the AST import allowlist both need real files to
mean anything. Nothing in this file touches the network, LM Studio, or `modelbench.lmstudio` /
`modelbench.tooling` (both separate, concurrent S2 units — the AST-check fixtures that name
`modelbench.tooling` are parsed, never imported).
"""

from __future__ import annotations

import hashlib
import json

import pytest
from conftest import pack_fixture

from modelbench.packs import (
    PackConfigError,
    content_hash,
    derive_call_surface,
    load_pack,
    pack_ref_from_manifest,
    validate_pack,
)

# --------------------------------------------------------------------------------------------
# load_pack / Pack
# --------------------------------------------------------------------------------------------


def test_load_pack_reads_identity_fields_from_the_manifest() -> None:
    pack = load_pack(pack_fixture("valid"))
    assert pack.packId == "fixture-tool-caller"
    assert pack.packVersion == "1.0.0"
    assert pack.role == "tool-caller"
    assert pack.root == pack_fixture("valid")
    assert pack.manifest["scorer"] == "toolcalls"


def test_load_pack_raises_on_a_missing_pack_json(tmp_path) -> None:
    with pytest.raises(PackConfigError, match="pack.json is absent"):
        load_pack(tmp_path)


def test_load_pack_raises_on_invalid_json(tmp_path) -> None:
    (tmp_path / "pack.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(PackConfigError, match="not valid JSON"):
        load_pack(tmp_path)


def test_load_pack_raises_on_a_manifest_missing_identity_keys(tmp_path) -> None:
    (tmp_path / "pack.json").write_text(json.dumps({"packId": "p"}), encoding="utf-8")
    with pytest.raises(PackConfigError, match="packVersion, role"):
        load_pack(tmp_path)


def test_data_path_resolves_a_declared_data_key() -> None:
    pack = load_pack(pack_fixture("valid"))
    assert pack.data_path("conversations") == pack_fixture("valid") / "conversations.jsonl"


def test_data_path_raises_on_an_undeclared_key() -> None:
    pack = load_pack(pack_fixture("valid"))
    with pytest.raises(PackConfigError, match="data.nonexistent"):
        pack.data_path("nonexistent")


def test_load_tool_module_imports_the_pack_module_via_importlib() -> None:
    pack = load_pack(pack_fixture("valid"))
    module = pack.load_tool_module()
    assert module.build_environment()["ok"] is True


def test_load_tool_module_raises_when_the_manifest_declares_no_module(tmp_path) -> None:
    (tmp_path / "pack.json").write_text(
        json.dumps({"packId": "p", "packVersion": "1.0.0", "role": "guard-judge"}),
        encoding="utf-8",
    )
    pack = load_pack(tmp_path)
    with pytest.raises(PackConfigError, match="tools.module is absent"):
        pack.load_tool_module()


# --------------------------------------------------------------------------------------------
# content_hash (§5 test 4)
# --------------------------------------------------------------------------------------------


def test_content_hash_changes_on_any_byte_and_ignores_provenance(tmp_path) -> None:
    root = tmp_path / "pack"
    root.mkdir()
    (root / "a.txt").write_text("alpha")
    (root / "b.txt").write_text("beta")
    (root / "PROVENANCE.md").write_text("origin notes")
    baseline = content_hash(root)

    # unchanged when PROVENANCE.md changes
    (root / "PROVENANCE.md").write_text("a completely different provenance note")
    assert content_hash(root) == baseline

    # changes when any byte in any pack file changes
    (root / "b.txt").write_text("BETA")
    assert content_hash(root) != baseline


def test_content_hash_is_independent_of_directory_listing_order(tmp_path) -> None:
    """`content_hash` sorts paths itself rather than trusting the filesystem's own listing order.

    A plain two-file a/b case does not distinguish this: on this filesystem (confirmed directly
    with `os.listdir`), `Path.rglob` already happens to return two files alphabetically regardless
    of creation order. Five files whose names are **not** alphabetical (`zeta`, `mike`, `alpha`,
    `delta`, `charlie` — the exact sequence `os.listdir` was observed to return unsorted on this
    box) is what actually exercises the sort: hand-compute the expected hash by sorting the same
    five names in Python and assert `content_hash` agrees, so a removed `sorted()` call — which
    would follow the filesystem's own (here, non-alphabetical) order — fails this test."""
    root = tmp_path / "pack"
    root.mkdir()
    names = ["zeta", "mike", "alpha", "delta", "charlie"]
    for name in names:
        (root / f"{name}.txt").write_text(name)

    expected = hashlib.sha256()
    for name in sorted(f"{n}.txt" for n in names):
        expected.update(name.encode("utf-8"))
        expected.update(b"\0")
        expected.update(name.removesuffix(".txt").encode("utf-8"))
        expected.update(b"\0")

    assert content_hash(root) == expected.hexdigest()


def test_content_hash_excludes_pycache(tmp_path) -> None:
    """A bytecode cache `load_tool_module` (or any import machinery) might leave behind is not
    pack content — changing it must not change the pack's identity."""
    root = tmp_path / "pack"
    (root / "__pycache__").mkdir(parents=True)
    (root / "a.txt").write_text("alpha")
    (root / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\x00\x01")
    baseline = content_hash(root)

    (root / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\x02\x03\x04")
    assert content_hash(root) == baseline


def test_content_hash_matches_a_pack_with_one_byte_changed(tmp_path) -> None:
    """Sanity check for the exact algorithm read out directly: SHA-256 over sorted relative paths
    and file bytes, both NUL-delimited — proven independent of the implementation's internals by
    reproducing it from first principles over a single-file pack."""
    root = tmp_path / "pack"
    root.mkdir()
    (root / "only.txt").write_text("hello")
    expected = hashlib.sha256()
    expected.update(b"only.txt\0hello\0")
    assert content_hash(root) == expected.hexdigest()


# --------------------------------------------------------------------------------------------
# The §3.3 totality boundary — asserted, not assumed
# --------------------------------------------------------------------------------------------


def test_loaded_pack_ref_content_hash_is_never_none_and_matches_content_hash() -> None:
    root = pack_fixture("valid")
    pack = load_pack(root)
    ref = pack.ref()
    assert ref.contentHash is not None
    assert ref.contentHash == content_hash(root)


def test_manifest_only_pack_ref_content_hash_is_none() -> None:
    ref = pack_ref_from_manifest(pack_fixture("valid") / "pack.json")
    assert ref.contentHash is None


# --------------------------------------------------------------------------------------------
# validate_pack — the sampling contract (§3.3)
# --------------------------------------------------------------------------------------------


def test_validate_pack_accepts_a_fully_valid_pack() -> None:
    pack = load_pack(pack_fixture("valid"))
    assert validate_pack(pack) == []


def test_validate_pack_rejects_the_row_count_identity_specifically() -> None:
    """A pack declaring `analysisUnit: "conversationId"` under `scripts: 12,
    replicatesPerScript: 4` — 48 distinct `conversationId` values where 12 are required (plan §3.3,
    §4 S2). `pairingKey[0] == analysisUnit`, so the structural route passes; only the row-count
    identity catches this."""
    pack = load_pack(pack_fixture("row_count_violation"))
    problems = validate_pack(pack)
    # This fixture's `replicatesPerScript: 4` also trips Rule 6 (below) — the row-count identity
    # is asserted specifically, by content, rather than by assuming it is the only problem.
    row_count = [p for p in problems if "row-count identity" in p]
    assert len(row_count) == 1
    assert "48 distinct" in row_count[0]
    assert "sampling.scripts=12" in row_count[0]


def test_validate_pack_rejects_undeclared_replication_row_count_only() -> None:
    """The case §3.3 names as the row-count identity's *reason to exist*: a pack declaring
    `replicatesPerScript: 1` that ships four conversations per script — "the case that slips past
    Rule 6's declaration check and past Rule 1 at once" (plan §3.3). `replicatesPerScript == 1`
    so Rule 6 does not fire, `analysisUnit == pairingKey[0]` so the structural route passes; only
    the data-driven row-count identity — reading `data.conversations`, the plan's own manifest key
    (line 435) — catches it."""
    pack = load_pack(pack_fixture("undeclared_replication"))
    problems = validate_pack(pack)
    assert not any("Rule 6" in p for p in problems)
    row_count = [p for p in problems if "row-count identity" in p]
    assert len(row_count) == 2
    assert any("holds 48 rows, expected" in p and "12 × 1 = 12" in p for p in row_count)
    assert any(
        "do not each appear sampling.replicatesPerScript=1 times" in p for p in row_count
    )


def test_validate_pack_rejects_analysis_unit_outside_pairing_key_structurally() -> None:
    """`analysisUnit` declared outside its own `pairingKey[0]` — caught by
    `check_sampling_contract`, reused rather than re-implemented (plan §3.3, impl review Pass 1
    §4 item 6)."""
    pack = load_pack(pack_fixture("analysis_unit_outside_pairing_key"))
    problems = validate_pack(pack)
    assert problems == [
        "sampling.analysisUnit 'conversationId' is not pairingKey[0] 'scriptId'; the analysis "
        "unit is the outermost component of the pairing key, by rule (plan §3.3)"
    ]


def test_validate_pack_rejects_replicates_per_script_greater_than_one() -> None:
    """`-ml` §3.4 Rule 6: only the one-level `cluster_bootstrap` exists, so a pack declaring
    `replicatesPerScript > 1` must fail validation rather than silently take the wrong interval."""
    pack = load_pack(pack_fixture("replicates_per_script_violation"))
    problems = validate_pack(pack)
    assert problems == [
        "fixture-replicates-per-script-violation: sampling.replicatesPerScript=2 > 1 is "
        "rejected; only the one-level cluster_bootstrap exists (-ml §3.4 Rule 6)"
    ]


def test_validate_pack_accepts_replicates_per_script_of_exactly_one() -> None:
    """The positive case beside Rule 6's rejection — `replicatesPerScript == 1` must not trip it,
    proving the check discriminates on `> 1` rather than on the field's mere presence."""
    pack = load_pack(pack_fixture("valid"))
    assert not any("Rule 6" in problem for problem in validate_pack(pack))


# --------------------------------------------------------------------------------------------
# validate_pack — callSurface (§3.4.4a)
# --------------------------------------------------------------------------------------------


def test_derive_call_surface_maps_each_requirement_token() -> None:
    assert derive_call_surface(["lmstudio-chat"]) == "chat"
    assert derive_call_surface(["lmstudio-embeddings"]) == "embeddings"


def test_derive_call_surface_is_none_on_neither_or_both() -> None:
    assert derive_call_surface([]) is None
    assert derive_call_surface(["lmstudio-chat", "lmstudio-embeddings"]) is None


def test_validate_pack_rejects_a_pack_declaring_neither_call_surface() -> None:
    pack = load_pack(pack_fixture("call_surface_neither"))
    problems = validate_pack(pack)
    assert problems == [
        "fixture-call-surface-neither: environment.requires must declare exactly one of "
        "'lmstudio-chat' / 'lmstudio-embeddings' (plan §3.4.4a); found []"
    ]


def test_validate_pack_rejects_a_pack_declaring_both_call_surfaces() -> None:
    pack = load_pack(pack_fixture("call_surface_both"))
    problems = validate_pack(pack)
    assert problems == [
        "fixture-call-surface-both: environment.requires must declare exactly one of "
        "'lmstudio-chat' / 'lmstudio-embeddings' (plan §3.4.4a); "
        "found ['lmstudio-chat', 'lmstudio-embeddings']"
    ]


def test_validate_pack_accepts_a_pack_declaring_exactly_one_call_surface() -> None:
    pack = load_pack(pack_fixture("valid"))
    assert not any("environment.requires" in problem for problem in validate_pack(pack))


# --------------------------------------------------------------------------------------------
# validate_pack — the AST import allowlist (§3.3)
# --------------------------------------------------------------------------------------------


def test_validate_pack_rejects_a_module_importing_outside_the_allowlist() -> None:
    pack = load_pack(pack_fixture("bad_import"))
    problems = validate_pack(pack)
    assert problems == [
        "fixture-bad-import: tools/sim.py imports from 'not_an_allowed_package', which is "
        "outside the pack import allowlist (stdlib + 'modelbench.tooling', plan §3.3)"
    ]


def test_validate_pack_accepts_a_module_importing_modelbench_tooling() -> None:
    """`modelbench.tooling` is the one non-stdlib import a pack module may make (plan §3.3). This
    pack is never executed — `modelbench.tooling` does not exist yet — only AST-walked."""
    pack = load_pack(pack_fixture("tooling_import_allowed"))
    assert validate_pack(pack) == []


def test_validate_pack_accepts_the_valid_packs_stdlib_only_module() -> None:
    pack = load_pack(pack_fixture("valid"))
    assert not any("allowlist" in problem for problem in validate_pack(pack))
