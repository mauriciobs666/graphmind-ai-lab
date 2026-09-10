"""§4 S2's real pack loader — `load_pack`, `content_hash`, `validate_pack`, and the §3.3 totality
boundary (plan §4 S2, §5 test 4 and test 12).

Every pack here is a real directory — `tests/fixtures/packs/` (`conftest.pack_fixture`) for the
checked-in shapes, `tmp_path` for the row-count identity's coverage probe, which needs a small
combinatorial family of manifests rather than one fixture each. Both are real files on disk, never
an in-memory manifest: the row-count identity and the AST import allowlist both need real files to
mean anything. Nothing in this file touches the network, LM Studio, or `modelbench.lmstudio` /
`modelbench.tooling` (both separate, concurrent S2 units — the AST-check fixtures that name
`modelbench.tooling` are parsed, never imported).
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest
from conftest import pack_fixture

from modelbench.packs import (
    _ROW_COUNT_IDENTITY_KEY_HINTS,
    _STDLIB_MODULE_NAMES,
    ROW_COUNT_IDENTITY_EXEMPT_CELLS,
    ROW_COUNT_IDENTITY_KEYS,
    PackConfigError,
    PackMetrics,
    PackRef,
    _row_count_identity_field_valid,
    _row_count_identity_problems,
    check_sampling_contract,
    content_hash,
    derive_call_surface,
    load_pack,
    pack_ref_from_manifest,
    validate_pack,
)
from modelbench.roles import ROLES, analysis_unit_field

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


def test_validate_pack_rejects_a_scripts_declaring_pack_missing_data_conversations() -> None:
    """A pack that declares `sampling.scripts` — conversation-shaped, per §3.3's own manifest
    literal and the "conversation pack" run-shape rule (plan §3.3, §3.9 point 2) — but has no
    `data` block at all, so `data.conversations` is absent. The docstring's stated exemption is
    for a pack of a *different shape* — an item-level pack, which declares no `scripts` either —
    not for a conversation-shaped pack that simply forgot its rows file. Only the row-count
    identity route can tell these two "no `data.conversations`" cases apart, since the structural
    route (`analysisUnit == pairingKey[0]`) does not look at `data` at all and passes either way."""
    pack = load_pack(pack_fixture("missing_data_conversations"))
    problems = validate_pack(pack)
    assert problems == [
        "fixture-missing-data-conversations: data.conversations must be a non-empty string path "
        "— a scripts-declaring pack is conversation-shaped and needs a rows file to check the "
        "identity against, got NoneType (None) (row-count identity, plan §3.3)"
    ]


# --------------------------------------------------------------------------------------------
# validate_pack — the row-count identity's own coverage (impl review Pass 12, P12-6, §4A)
# --------------------------------------------------------------------------------------------

#: Rows that violate `scripts: 12, replicatesPerScript: 1` — 48 rows, 12 distinct `scriptId`
#: values each appearing 4 times where 1 is required — the same shape Appendix L.5 built the
#: coverage table against.
_VIOLATING_ROWS = [{"scriptId": f"script-{i % 12:02d}", "turnIndex": 0} for i in range(48)]

#: Rows that satisfy `scripts: 12, replicatesPerScript: 1` exactly — the all-valid control's rows.
_SATISFYING_ROWS = [{"scriptId": f"script-{i:02d}", "turnIndex": 0} for i in range(12)]

_ROW_COUNT_IDENTITY_VALUE_KINDS = ("valid", "absent", "wrong-type")

#: One concrete wrong-type value per key. `sampling.analysisUnit` and `data.conversations` use a
#: plain int rather than an unhashable type (a list would make `row.get(analysis_unit)` raise
#: `TypeError: unhashable type`, which is a different failure than the one this probe is for).
_ROW_COUNT_IDENTITY_WRONG_TYPE_VALUES = {
    "sampling.scripts": "12",
    "sampling.replicatesPerScript": "1",
    "sampling.analysisUnit": 123,
    "data.conversations": 123,
}


def _row_count_coverage_manifest(pack_id: str) -> dict:
    return {
        "packId": pack_id,
        "packVersion": "1.0.0",
        "role": "tool-caller",
        "environment": {"requires": ["lmstudio-chat"]},
        "data": {"conversations": "conversations.jsonl"},
        "sampling": {
            "scripts": 12,
            "replicatesPerScript": 1,
            "seed": 20260909,
            "pairingKey": ["scriptId", "turnIndex"],
            "analysisUnit": "scriptId",
        },
        "metrics": {
            "verdictMetrics": ["cleanThroughTurnH"],
            "headlineMetric": "cleanThroughTurnH",
        },
    }


def _apply_row_count_cell(manifest: dict, key: str, kind: str) -> dict:
    """Perturb `manifest` at `key` (one of `ROW_COUNT_IDENTITY_KEYS`) to `kind`
    ("valid" / "absent" / "wrong-type"), leaving every other key untouched."""
    manifest = copy.deepcopy(manifest)
    block, name = key.split(".", 1)
    target = manifest["sampling"] if block == "sampling" else manifest.setdefault("data", {})
    if kind == "valid":
        pass
    elif kind == "absent":
        target.pop(name, None)
    elif kind == "wrong-type":
        target[name] = _ROW_COUNT_IDENTITY_WRONG_TYPE_VALUES[key]
    else:
        raise ValueError(kind)
    return manifest


def _write_row_count_pack(root: Path, manifest: dict, rows: list[dict]):
    root.mkdir(parents=True)
    (root / "pack.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "conversations.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return load_pack(root)


def test_row_count_identity_coverage_over_its_own_keys_and_value_kinds(tmp_path) -> None:
    """impl review Pass 12 §4A / Pass 13 P13-2: the four manifest keys
    `_row_count_identity_problems` actually reads (`packs.ROW_COUNT_IDENTITY_KEYS`) crossed with
    three value-kinds — plan-valid, absent, present but not the declared type — over a rows file
    that violates the identity, plus an all-valid control against a rows file that does not.
    Every cell must either report at least one problem **from the route itself** or be named in
    `packs.ROW_COUNT_IDENTITY_EXEMPT_CELLS`; this probe calls `_row_count_identity_problems`
    directly — not `validate_pack` — and computes the actual silent cells by execution, asserting
    that set equals the exemption constant exactly, so a stale exemption is as loud as a newly
    silent one. **Pass 13 P13-2:** `validate_pack(pack) == []` measures the whole validator's
    silence, not the row-count route's — `_sampling_problems`' other call, `pack.ref()`, can flag
    the same manifest for a reason that has nothing to do with the row-count arithmetic (e.g.
    `_ref_from_manifest_fields` refusing a manifest with no `sampling.analysisUnit` at all, before
    `check_sampling_contract` even runs) and keep `validate_pack(pack)` non-empty while
    `_row_count_identity_problems` itself is silent underneath — so a fourth round of a
    P12-6-shaped silent branch never registers as newly silent under the old criterion. Calling the
    route directly is what closes that gap. The axis list is `ROW_COUNT_IDENTITY_KEYS` itself, the
    same constant the route consults to know which manifest fields it owns — a fifth key added
    there is probed automatically, without a second edit here.

    Written against the plan (§3.3's row-count identity), not against the implementation: run
    against the pre-P12-6-fix predicate, this must go red on at least
    `("sampling.scripts", "wrong-type")`, `("sampling.replicatesPerScript", "absent")` and
    `("sampling.replicatesPerScript", "wrong-type")` — the three cells P12-6 names silent; and
    against a reinserted P12-6-shaped branch (`if "analysisUnit" not in sampling: return []` at
    the route's head), it must go red on `("sampling.analysisUnit", "absent")` — even though
    `validate_pack` does **not** stay `[]` on that cell (`_ref_from_manifest_fields` independently
    refuses a manifest with no `sampling.analysisUnit` at all, before `check_sampling_contract`
    ever runs), which is exactly the old probe's blind spot: `validate_pack(pack) == []` was never
    true on this cell, so it was never counted "silent" even while the route itself, the mechanism
    the probe is named for, had gone silent underneath an unrelated problem that happened to cover
    the same manifest (Pass 13 P13-2, verified directly, not from the review's own appendix)."""
    base = _row_count_coverage_manifest("fixture-row-count-coverage")

    silent_cells: set[tuple[str, str]] = set()
    for key in ROW_COUNT_IDENTITY_KEYS:
        for kind in _ROW_COUNT_IDENTITY_VALUE_KINDS:
            manifest = _apply_row_count_cell(base, key, kind)
            manifest["packId"] = f"fixture-row-count-coverage-{key}-{kind}"
            pack = _write_row_count_pack(
                tmp_path / f"cell-{key}-{kind}", manifest, _VIOLATING_ROWS
            )
            sampling = pack.manifest.get("sampling") or {}
            if _row_count_identity_problems(pack, sampling) == []:
                silent_cells.add((key, kind))

    assert silent_cells == set(ROW_COUNT_IDENTITY_EXEMPT_CELLS)

    control_manifest = _row_count_coverage_manifest("fixture-row-count-coverage-control")
    control_pack = _write_row_count_pack(
        tmp_path / "cell-control", control_manifest, _SATISFYING_ROWS
    )
    control_sampling = control_pack.manifest.get("sampling") or {}
    assert _row_count_identity_problems(control_pack, control_sampling) == []
    assert validate_pack(control_pack) == []


def test_every_row_count_identity_key_has_a_hint_and_no_hint_has_no_key() -> None:
    """`_ROW_COUNT_IDENTITY_KEY_HINTS` is a second declaration of `ROW_COUNT_IDENTITY_KEYS`, and
    nothing bound them (impl review Pass 16, P16-3): dropping a hint reddened only because
    `_row_count_identity_field_problem` `KeyError`s on a key some other test happened to
    perturb, and **adding** a hint for a key that does not exist left the suite green — an
    entry no message can ever render, and the shape a fifth key added to one table alone takes.
    """
    assert set(_ROW_COUNT_IDENTITY_KEY_HINTS) == set(ROW_COUNT_IDENTITY_KEYS)


def test_each_row_count_identity_keys_own_hint_reaches_its_own_message(tmp_path) -> None:
    """The other direction of the same pin, per member: the hint written for a key is the hint
    the operator is shown when *that* key is the one that is wrong.

    This pins the *lookup*, not the wording: it reads the hint from the same table the message
    does, so swapping two hints between keys leaves it green (measured). The wording is pinned
    against the predicate in the next test instead.
    """
    base = _row_count_coverage_manifest("fixture-row-count-hints")
    for key in ROW_COUNT_IDENTITY_KEYS:
        manifest = _apply_row_count_cell(base, key, "wrong-type")
        manifest["packId"] = f"fixture-row-count-hints-{key}"
        pack = _write_row_count_pack(tmp_path / f"hint-{key}", manifest, _VIOLATING_ROWS)
        problems = _row_count_identity_problems(pack, pack.manifest["sampling"])
        assert problems, f"{key} wrong-type produced no problem to carry a hint"
        assert f"{key} must be {_ROW_COUNT_IDENTITY_KEY_HINTS[key]}" in problems[0]


def test_each_hint_states_the_type_its_own_validity_predicate_enforces() -> None:
    """The hint's *prose*, bound to the behaviour it claims to describe.

    The table's domain and its lookup are both pinned above, and neither can see a hint pasted
    under the wrong key: swapping `sampling.scripts`' hint with `sampling.analysisUnit`'s leaves
    a domain-correct table that tells the operator to make an int a string, and the whole suite
    stayed green on exactly that mutation. So the two claims the hints make are asserted against
    what `_row_count_identity_field_valid` actually accepts, per key. The two prefixes below are
    the test's own literals — reading them off the table is what made the swap invisible.
    """
    for key in ROW_COUNT_IDENTITY_KEYS:
        accepts_int = _row_count_identity_field_valid(key, 12)
        accepts_str = _row_count_identity_field_valid(key, "twelve")
        assert accepts_int is not accepts_str, f"{key} accepts both types or neither"
        expected = "a plain int" if accepts_int else "a non-empty string"
        assert _ROW_COUNT_IDENTITY_KEY_HINTS[key].startswith(expected), (
            f"{key}: hint {_ROW_COUNT_IDENTITY_KEY_HINTS[key]!r} does not describe the type "
            f"its own predicate enforces"
        )


def test_row_count_identity_exempt_cells_each_carry_a_reason() -> None:
    """The exemption constant's whole point is a one-line reason beside each sanctioned silent
    cell (plan §4A) — an empty or missing reason defeats that, silently."""
    assert ROW_COUNT_IDENTITY_EXEMPT_CELLS
    for cell, reason in ROW_COUNT_IDENTITY_EXEMPT_CELLS.items():
        assert isinstance(cell, tuple) and len(cell) == 2
        assert cell[0] in ROW_COUNT_IDENTITY_KEYS
        assert isinstance(reason, str) and reason.strip()


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


# --------------------------------------------------------------------------------------------
# check_sampling_contract — route (iii): the role's own analysis-unit field (v1.25)
#
# Routes (i) and (ii) are both satisfied by any *self-consistent* naming (plan §3.3): a pack that
# declares `analysisUnit == pairingKey[0]` and a row count that matches passes both while still
# naming the wrong field for its role. Route (iii) is the only one that catches that — so it must
# be driven by *executing* `check_sampling_contract`, never by reading
# `roles.ANALYSIS_UNIT_FIELD_BY_ROLE` back (plan §4 S2 "Done when", `:5078-5082`).
# --------------------------------------------------------------------------------------------

#: Per role, a `pairingKey[0]` value that is self-consistent with route (i)
#: (`analysisUnit == pairingKey[0]`) while violating route (iii) alone — the "consistently wrong
#: id" that neither route (i) nor route (ii) can catch (plan §3.3). `tool-caller`'s is P12-7's own
#: fixture shape (plan §4 S2 "Done when", `:5078-5082`): `"conversationId"`, the exact value that
#: shipped as a positive control before U76's fix — not merely "some other role's field", since
#: `"conversationId"` is not any role's own `analysis_unit_field` at all. Every other role borrows
#: `"scriptId"` — `tool-caller`'s own field, a real row's value, never the borrowing role's own.
_ROUTE_III_WRONG_PAIRING_KEY_HEAD: dict[str, str] = {
    "tool-caller": "conversationId",
    "guard-judge": "scriptId",
    "nlq-generator": "scriptId",
    "chat-responder": "scriptId",
    "embedder": "scriptId",
}


def _sampling_ref(role: str, pairing_key_head: str) -> PackRef:
    """A minimal `PackRef` isolating route (iii): `pairingKey`/`analysisUnit` agree with each
    other, so route (i) passes, and `check_sampling_contract` never reads pack data at all, so
    there is no row-count identity (route (ii)) in scope to fail or pass here — only route (iii)
    is exercised (plan §3.3)."""
    return PackRef(
        packId=f"route-iii-{role}",
        packVersion="1.0.0",
        contentHash=None,
        role=role,
        metrics=PackMetrics(verdictMetrics=("m",), headlineMetric="m"),
        pairingKey=(pairing_key_head,),
        analysisUnit=pairing_key_head,
        seed=20260909,
    )


@pytest.mark.parametrize("role", ROLES)
def test_check_sampling_contract_rejects_a_pairing_key_head_outside_the_roles_own_field(
    role,
) -> None:
    """Route (iii), v1.25: `pairingKey[0]` must equal `roles.analysis_unit_field(role)`, swept
    over **every** member of `roles.ROLES` by actually calling `check_sampling_contract` — not by
    reading `ANALYSIS_UNIT_FIELD_BY_ROLE` back (plan §4 S2 "Done when", `:5078-5082`)."""
    wrong = _ROUTE_III_WRONG_PAIRING_KEY_HEAD[role]
    assert wrong != analysis_unit_field(role), (
        f"fixture bug: {wrong!r} is already role {role!r}'s own field, so this case would not "
        "isolate route (iii)"
    )
    ref = _sampling_ref(role, wrong)
    with pytest.raises(PackConfigError, match=re.escape(wrong)):
        check_sampling_contract(ref)


def test_check_sampling_contract_tool_caller_route_iii_case_is_p12_7s_fixture_shape() -> None:
    """The `tool-caller` sweep case, named explicitly in the plan (`:5078-5082`):
    `pairingKey[0] == "conversationId"` — P12-7's own historical defect fixture (corrected at
    U76). It is self-consistent (`analysisUnit == pairingKey[0]`, route (i) passes) and is caught
    only by route (iii), which this test isolates by name rather than folding it anonymously into
    the sweep above."""
    ref = _sampling_ref("tool-caller", "conversationId")
    with pytest.raises(PackConfigError, match="analysis-unit field"):
        check_sampling_contract(ref)


@pytest.mark.parametrize("role", ROLES)
def test_check_sampling_contract_accepts_the_roles_own_field(role) -> None:
    """The positive control beside the sweep above: each role's own `analysis_unit_field(role)`
    as `pairingKey[0]` must not trip route (iii)."""
    ref = _sampling_ref(role, analysis_unit_field(role))
    assert check_sampling_contract(ref) is None


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


def test_the_pack_import_allowlist_is_the_runtime_registry_with_nothing_added() -> None:
    """`_STDLIB_MODULE_NAMES` is the allowlist half that decides what a pack module may import,
    and it was the one constant this component held out as unpinnable (impl review Pass 16,
    P16-5, now withdrawn). Measured before writing this: appending one name —
    `frozenset(sys.stdlib_module_names) | {"requests"}` — silently widened what every pack is
    permitted to import and left the suite at 940 passed.

    **The equality is not circular, and that is the whole finding.** Binding a derived constant
    to its own source looks tautological, and against a *re-derivation* it is: rewriting the
    constant as an equivalent comprehension stays green, correctly, and should. What
    it refuses is an **augmentation** — `<derived> | {extra}`, the shape a hand-added exception
    takes — which no other assertion in this file can see, because the guard's behavioural tests
    below refuse `not_an_allowed_package` just as happily with `requests` admitted, and a
    per-member consequence would be one assertion per stdlib module, ~300 of them.

    So the reach claim in the constant's own docstring — *"no hand-maintained list to fall out of
    date"* — is asserted here as what it actually says: derived, and **only** derived.
    """
    assert _STDLIB_MODULE_NAMES == frozenset(sys.stdlib_module_names)


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


# --------------------------------------------------------------------------------------------
# validate_pack — tools.module's own path (impl review Pass 12, P12-2 / P12-3(i))
# --------------------------------------------------------------------------------------------


def test_validate_pack_rejects_a_tool_module_outside_the_pack_root() -> None:
    """`"tools": {"module": "../outside.py"}` validated CLEAN before this fix, `load_tool_module`
    executed the outside file, and `content_hash` never moved when that file changed — falsifying
    §3.3's "pack code is part of the content hash" (impl review Appendix L.2). The fix is at
    `validate_pack`, not at `load_tool_module`: report it rather than let it raise."""
    pack = load_pack(pack_fixture("tools_module_outside_root"))
    problems = validate_pack(pack)
    assert problems == [
        "fixture-tools-module-outside-root: tools.module '../outside.py' resolves outside the "
        "pack root; pack code must live under the pack directory so it is covered by content_hash "
        "(plan §3.3, impl review P12-2)"
    ]


def test_validate_pack_rejects_a_non_py_tool_module() -> None:
    """`tools/sim.pyc` as `tools.module` validated CLEAN before this fix and the unscanned
    bytecode executed — the AST walk globs `*.py` only (impl review Appendix L.3(4)). A
    file-selection gap, not a syntactic-versus-semantic one: closed by requiring the `.py` suffix
    `validate_pack` can actually scan."""
    pack = load_pack(pack_fixture("tools_module_not_py"))
    problems = validate_pack(pack)
    assert problems == [
        "fixture-tools-module-not-py: tools.module 'tools/sim.pyc' must end in '.py'; the AST "
        "import allowlist only scans '*.py' files, so anything else loads unscanned (plan §3.3, "
        "impl review P12-3(i))"
    ]


def test_validate_pack_accepts_the_valid_packs_tool_module_path() -> None:
    """The positive case beside both new refusals — a `.py` module under the pack root must not
    trip either check."""
    pack = load_pack(pack_fixture("valid"))
    problems = validate_pack(pack)
    assert not any("tools.module" in problem for problem in problems)


# --------------------------------------------------------------------------------------------
# The "valid" fixture's own role shape (impl review Pass 12, P12-7)
# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
# validate_pack — the `prompt` block: `historyReplay` and `maxIterationsPerTurn`'s role scoping
# (plan §3.3, v1.26; impl review Pass 17, P17-5)
# --------------------------------------------------------------------------------------------


def test_validate_pack_rejects_a_bad_history_replay_value() -> None:
    """A `historyReplay` value outside `convo._HISTORY_REPLAY_MODES` must fail at validate time,
    not only inside `convo.assemble` at first use (impl review P17-5)."""
    pack = load_pack(pack_fixture("prompt_bad_history_replay"))
    problems = validate_pack(pack)
    assert problems == [
        "fixture-prompt-bad-history-replay: prompt.historyReplay 'verbose' is not one of "
        "['none', 'plaintext', 'structured', 'structured-replies-only'] (plan §3.3)"
    ]


def test_validate_pack_rejects_a_missing_required_max_iterations_per_turn() -> None:
    """`tool-caller` is `roles.MULTI_CALL_TURN_BY_ROLE`'s one `True` member, so
    `prompt.maxIterationsPerTurn` is required on this pack; omitting it must fail validation
    rather than only `drive`'s own `None` refusal at first use."""
    pack = load_pack(pack_fixture("prompt_missing_max_iterations"))
    problems = validate_pack(pack)
    assert problems == [
        "fixture-prompt-missing-max-iterations: prompt.maxIterationsPerTurn is absent; role "
        "'tool-caller' runs a multi-call turn and the field is required (plan §3.3, v1.26)"
    ]


def test_validate_pack_rejects_a_forbidden_max_iterations_per_turn() -> None:
    """`guard-judge` is single-call by construction (`roles.MULTI_CALL_TURN_BY_ROLE['guard-judge']
    is False`), so `prompt.maxIterationsPerTurn` is forbidden on this pack — the direction P17-5
    named as untested alongside the missing-but-required one."""
    pack = load_pack(pack_fixture("prompt_forbidden_max_iterations"))
    problems = validate_pack(pack)
    assert problems == [
        "fixture-prompt-forbidden-max-iterations: prompt.maxIterationsPerTurn is present but "
        "role 'guard-judge' is not multi-call, where the field is forbidden (plan §3.3, v1.26)"
    ]


def test_prompt_config_resolves_the_valid_packs_prompt_block() -> None:
    """The happy path: a well-formed `prompt` block validates clean (already asserted by
    `test_validate_pack_accepts_a_fully_valid_pack`) and the resulting `PromptConfig` is
    available from `Pack.prompt_config()` — the accessor downstream code (a future runner unit)
    reaches for rather than re-parsing the manifest itself."""
    pack = load_pack(pack_fixture("valid"))
    cfg = pack.prompt_config()
    assert cfg.historyReplay == "structured"
    assert cfg.representToolSchemasEachTurn is True
    assert cfg.historyTurns == 0
    assert cfg.maxIterationsPerTurn == 8
    assert cfg.temperature == 0.0
    assert cfg.maxTokens == 1024


def test_prompt_config_is_not_this_route_when_the_manifest_has_no_prompt_block() -> None:
    """A pack with no `prompt` key at all (every fixture but `valid`, today) must not collect a
    spurious `prompt`-related problem — mirroring `_tool_module_problems`'s "absent is not this
    function's problem" convention for `tools.module`."""
    pack = load_pack(pack_fixture("call_surface_both"))
    assert not any("prompt" in problem for problem in validate_pack(pack))


def test_the_valid_fixtures_analysis_unit_is_scriptId_not_a_conversation_id() -> None:
    """§3.3: "the analysis unit is the *outermost* component of `pairingKey` … For the
    tool-caller that is `scriptId`, never a conversation id" — and the plan's own manifest literal
    declares `pairingKey: ["scriptId", ...]`. The fixture previously declared
    `role: "tool-caller"` with `analysisUnit: "conversationId"`: it satisfied the *mechanised*
    half of the rule (`analysisUnit == pairingKey[0]`) while violating the *stated* half, which
    made it a positive control written against the implementation rather than the plan (impl
    review P12-7)."""
    pack = load_pack(pack_fixture("valid"))
    assert pack.manifest["role"] == "tool-caller"
    ref = pack.ref()
    assert ref.analysisUnit == "scriptId"
    assert ref.pairingKey[0] == "scriptId"
