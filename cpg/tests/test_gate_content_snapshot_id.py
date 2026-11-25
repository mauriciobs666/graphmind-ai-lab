"""The dev-gold probe vouches for a CONTENT id, not the stamped snapshot id (``docs/reviews/connected-code-graph-p1-gate-snapshot-rca.md`` §5 (b)).

The stamped ``snapshotId`` covers per-unit ``bundleHash``es, and 182 units carry HMAC-keyed fields (``valueHash``, ``defaultHash``, ``bindParts``,
``bindKeyCheck``) keyed by a random key per build, so two builds of one commit never share it. ``G.content_snapshot_id`` is the same identity over the same
inputs with those four fields left out of every bundle hash. Every expected value is a comparison between two builds, never a constant copied from the code.
"""
import copy
import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "eval" / "tools"))
import test_assemble_rows as TA  # noqa: E402

import gate as G  # noqa: E402
from ccg import atrest, valuehash  # noqa: E402
from ccg.assemble import snapshot as SN  # noqa: E402

SYSTEM = TA.SYSTEM
KEYED = ("valueHash", "defaultHash", "bindParts", "bindKeyCheck")


def keyed_world(keys_root: Path) -> list[list[dict]]:
    """``TA.world()`` with every keyed field recomputed under the system key of ``keys_root`` (a fresh random key per root, as a gate attempt gets)."""
    key = atrest.hmac_key(SYSTEM, keys_root)
    out = copy.deepcopy(TA.world())
    for records in out:
        for r in records:
            if r["type"] == "header" and "bindKeyCheck" in r:
                r["bindKeyCheck"] = atrest.bind_key_check(key)
            elif r["type"] == "configEntry":
                r["valueHash"] = valuehash.entry_hmac(key, "t", r["keyPath"])
                if "bindParts" in r:
                    r["bindParts"]["hostHmac"] = valuehash.entry_hmac(key, valuehash.BIND_HOST_TAG, r["bindParts"]["host"])
            elif r["type"] == "launchFact" and "defaultHash" in r:
                r["defaultHash"] = valuehash.entry_hmac(key, "t", r["default"])
    return out


def units_of(rows, bundles) -> list[list]:
    """``[unitId, status, records, flowAvailable, failReason]`` of the fresh units, read from the built rows (what the store's ``:Unit`` nodes carry)."""
    by_id = {b[0]["unitId"]: b for b in bundles}
    return [[u["unitId"], u["buildStatus"], by_id[u["unitId"]], u["flowAvailable"], u.get("failReason", "")] for u in rows.nodes["Unit"] if u["unitId"] in by_id]


def two_builds(tmp_path):
    out = []
    for n in (1, 2):
        (tmp_path / f"b{n}").mkdir()
        bundles = keyed_world(tmp_path / f"keys{n}")
        out.append((bundles, TA.build(tmp_path / f"b{n}", bundles=bundles)))
    return out


def content(rows, bundles, **over) -> str:
    snap = rows.snapshot
    args = {"system": SYSTEM, "pack_set": snap["packSetHash"], "code": snap["codeHash"], "units": units_of(rows, bundles)}
    args.update(over)
    return G.content_snapshot_id(args["system"], args["pack_set"], args["code"], args["units"])


def test_the_stamped_snapshot_id_differs_across_two_keys_and_the_content_id_does_not(tmp_path):
    (b1, r1), (b2, r2) = two_builds(tmp_path)
    assert b1 != b2 and r1.snapshot["snapshotId"] != r2.snapshot["snapshotId"]          # the documented cause: a key changes the stamped id
    assert content(r1, b1) == content(r2, b2)                                          # the content id vouches for what the key does not touch


def test_the_content_id_is_not_the_stamped_snapshot_id_even_without_keyed_fields(tmp_path):
    (b1, r1), _ = two_builds(tmp_path)
    assert content(r1, b1) != r1.snapshot["snapshotId"]


@pytest.mark.parametrize("field", KEYED)
def test_each_keyed_field_alone_is_left_out_of_the_content_id(tmp_path, field):
    (b1, r1), _ = two_builds(tmp_path)
    changed = copy.deepcopy(b1)
    hit = 0
    for records in changed:
        for r in records:
            if field in r:
                if field == "bindParts":                                                       # only its two keyed members are excluded
                    r[field] = {**r[field], "hostHmac": "f" * 64, "pathHmacs": ["f" * 64] * len(r[field]["pathHmacs"])}
                else:
                    r[field] = "f" * 64
                hit += 1
    assert hit                                                                         # the fixture does carry the field
    units = [[u[0], u[1], next(c for c in changed if c[0]["unitId"] == u[0]), u[3], u[4]] for u in units_of(r1, b1)]
    assert content(r1, b1, units=units) == content(r1, b1)


def moved_by(tmp_path, edit) -> bool:
    """Whether ``edit(bindParts-bearing configEntry)`` moves the content id of the two-key fixture."""
    tmp_path = Path(tempfile.mkdtemp(dir=tmp_path))                                    # one build pair per call
    (b1, r1), _ = two_builds(tmp_path)
    changed = copy.deepcopy(b1)
    (entry,) = [r for recs in changed for r in recs if r["type"] == "configEntry" and "bindParts" in r]
    edit(entry)
    units = [[u[0], u[1], next(c for c in changed if c[0]["unitId"] == u[0]), u[3], u[4]] for u in units_of(r1, b1)]
    return content(r1, b1, units=units) != content(r1, b1)


@pytest.mark.parametrize("name,edit", [("port", lambda e: e["bindParts"].update(port=9999)), ("scheme", lambda e: e["bindParts"].update(scheme="https")),
                                        ("hostKind", lambda e: e["bindParts"].update(hostKind="loopback")), ("host", lambda e: e["bindParts"].update(host="other")),
                                        ("presence", lambda e: e.pop("bindParts"))])
def test_the_clear_members_and_the_presence_of_bind_parts_stay_in_the_content_id(tmp_path, name, edit):
    assert moved_by(tmp_path, edit)


def test_a_different_hostHmac_does_not_move_the_content_id(tmp_path):
    assert not moved_by(tmp_path, lambda e: e["bindParts"].update(hostHmac="f" * 64))


def test_the_number_of_path_segments_stays_in_the_content_id_but_their_hmac_values_do_not(tmp_path):
    n = []
    moved_by(tmp_path, lambda e: n.append(len(e["bindParts"]["pathHmacs"])))
    same = n[0]
    assert not moved_by(tmp_path, lambda e: e["bindParts"].update(pathHmacs=[f"{i:064x}" for i in range(same)]))         # other values, the same length
    assert moved_by(tmp_path, lambda e: e["bindParts"].update(pathHmacs=[f"{i:064x}" for i in range(same + 1)]))         # one more segment
    assert moved_by(tmp_path, lambda e: e["bindParts"].update(pathHmacs=[f"{i:064x}" for i in range(max(same - 1, 0))]) if same else e["bindParts"].update(pathHmacs=["a" * 64]))


@pytest.mark.parametrize("path,value", [(("header", "builtTree"), "c" * 40), (("header", "builtCommit"), "c" * 40), (("symbol", "qname"), "svc/a.py::renamed"),
                                        (("configEntry", "keyPath"), "services.web.environment.OTHER"), (("launchFact", "port"), 9000)])
def test_a_non_keyed_field_of_a_bundle_moves_the_content_id(tmp_path, path, value):
    (b1, r1), _ = two_builds(tmp_path)
    kind, field = path
    changed = copy.deepcopy(b1)
    done = False
    for records in changed:
        for r in records:
            if r["type"] == kind and not done and (field in r or kind == "header"):
                r[field], done = value, True
    assert done
    units = [[u[0], u[1], next(c for c in changed if c[0]["unitId"] == u[0]), u[3], u[4]] for u in units_of(r1, b1)]
    assert content(r1, b1, units=units) != content(r1, b1)


@pytest.mark.parametrize("what", ["pack_set", "code", "status", "flow", "fail", "system", "unit_set"])
def test_the_other_identity_inputs_stay_in_the_content_id(tmp_path, what):
    (b1, r1), _ = two_builds(tmp_path)
    units = units_of(r1, b1)
    over = {"pack_set": {"pack_set": "0" * 32}, "code": {"code": "0" * 64}, "system": {"system": "other"},
            "status": {"units": [[units[0][0], "stale-kept", *units[0][2:]], *units[1:]]},
            "flow": {"units": [[*units[0][:3], not units[0][3], units[0][4]], *units[1:]]},
            "fail": {"units": [[*units[0][:4], "bundle:invalid"], *units[1:]]},
            "unit_set": {"units": units[1:]}}[what]
    assert content(r1, b1, **over) != content(r1, b1)


def test_a_unit_without_a_bundle_contributes_an_empty_hash_as_the_stamped_id_does(tmp_path):
    (b1, r1), _ = two_builds(tmp_path)
    units = units_of(r1, b1)
    a = G.content_snapshot_id(SYSTEM, "p", "c", [*units, ["nm", "not-mapped", None, False, "manifest:never-built"]])
    b = G.content_snapshot_id(SYSTEM, "p", "c", units)
    assert a != b


# --- the probe check compares the content id -------------------------------------------------------------------------------------------------------------
def test_the_probe_check_accepts_an_equal_content_id_even_when_the_stamped_snapshot_id_differs():
    G.check_probe_snapshot({"snapshotId": "s" * 32, "contentSnapshotId": "c" * 32}, "c" * 32)


def test_the_probe_check_refuses_another_content_id_even_when_the_stamped_snapshot_id_would_match():
    with pytest.raises(G.Refusal, match="content"):
        G.check_probe_snapshot({"snapshotId": "c" * 32, "contentSnapshotId": "s" * 32}, "c" * 32)


@pytest.mark.parametrize("probe", [{"snapshotId": "s" * 32}, {"snapshotId": "s" * 32, "contentSnapshotId": ""}, {"contentSnapshotId": 5}])
def test_a_probe_without_a_usable_content_id_is_refused_with_a_text_that_says_how_to_get_one(probe):
    with pytest.raises(G.Refusal) as e:
        G.check_probe_snapshot(probe, "c" * 32)
    assert "no content snapshot id" in str(e.value) and "dev-probe" in str(e.value) and "rehearse again on this build" not in str(e.value)


def test_the_refusal_names_what_to_do_not_the_unfulfillable_rehearse_again_on_this_build():
    with pytest.raises(G.Refusal) as e:
        G.check_probe_snapshot({"contentSnapshotId": "a" * 32}, "b" * 32)
    assert "rehearse again on this build" not in str(e.value) and "commit" in str(e.value)


# --- reading the content id off a built store -----------------------------------------------------------------------------------------------------------
class FakeStore:
    def __init__(self, rows, snapshot=None):
        self._rows = rows
        s = rows.snapshot
        self._inputs = snapshot or {"snapshotId": s["snapshotId"], "manifestHash": s["manifestHash"], "packSetHash": s["packSetHash"], "codeHash": s["codeHash"]}

    def snapshot_inputs(self):
        return dict(self._inputs)

    def unit_rows(self):
        return [[u["unitId"], u["buildStatus"], u.get("builtCommit"), u["flowAvailable"], u.get("failReason")] for u in self._rows.nodes["Unit"]]


def reader_of(bundles):
    by_id = {b[0]["unitId"]: b for b in bundles}
    calls = []

    def read(system, unit, build, root):
        calls.append((system, unit, build, root))
        return copy.deepcopy(by_id[unit])
    read.calls = calls
    return read


def test_the_store_reader_hashes_the_bundles_of_the_work_root_and_agrees_with_the_pure_function(tmp_path):
    (b1, r1), (b2, r2) = two_builds(tmp_path)
    read = reader_of(b1)
    got = G.store_content_id(FakeStore(r1), SYSTEM, tmp_path / "work", read_bundle=read)
    assert got == content(r1, b1) == content(r2, b2)
    assert {c[0] for c in read.calls} == {SYSTEM} and {c[3] for c in read.calls} == {tmp_path / "work"}
    assert all(c[2] == "a" * 40 for c in read.calls)                                   # the build is the unit's builtCommit read from the store


def test_the_store_reader_refuses_bundles_that_are_not_the_ones_the_stored_snapshot_was_built_from(tmp_path):
    (b1, r1), (b2, _) = two_builds(tmp_path)
    with pytest.raises(G.HarnessError, match="stored snapshot"):
        G.store_content_id(FakeStore(r1), SYSTEM, tmp_path / "work", read_bundle=reader_of(b2))      # another key's bundles: the stamped id would not reproduce


# --- the store reader's queries and the production seam --------------------------------------------------------------------------------------------------
def test_the_store_snapshot_reads_the_identity_inputs_and_the_unit_rows_sorted_with_nulls_as_empty():
    seen = []

    def run(cypher, params):
        seen.append(cypher)
        if "Snapshot" in cypher:
            return [["sid", "man", "pack", "code"]]
        return [["u2", "fresh", "b" * 40, 1, None], ["u1", "not-mapped", None, False, "manifest:never-built"]]

    st = G.StoreSnapshot(run)
    assert st.snapshot_inputs() == {"snapshotId": "sid", "manifestHash": "man", "packSetHash": "pack", "codeHash": "code"}
    assert st.unit_rows() == [["u1", "not-mapped", None, False, "manifest:never-built"], ["u2", "fresh", "b" * 40, True, ""]]
    assert all("RO" not in q and q.startswith("MATCH") for q in seen)
    with pytest.raises(G.HarnessError):
        G.StoreSnapshot(lambda c, p: []).snapshot_inputs()


def test_the_production_seam_hands_the_store_system_and_work_root_to_the_store_reader(monkeypatch, tmp_path):
    import gate_run as R
    got = []
    monkeypatch.setattr(G, "store_content_id", lambda snap, system, work_root: got.append((snap, system, work_root)) or "id")
    assert R._store_content_id("S", tmp_path, "sys") == "id" and got == [("S", "sys", tmp_path)]


def test_the_default_bundle_reader_reads_through_the_at_rest_custody_read_and_the_validator(monkeypatch, tmp_path):
    from ccg import bundle
    calls = []
    monkeypatch.setattr(atrest, "read_bundle", lambda system, unit, build, root: calls.append((system, unit, build, root)) or b"raw")
    monkeypatch.setattr(bundle, "loads", lambda raw: [{"type": "header", "raw": raw}])
    assert G._stored_bundle("sys", "u", "b" * 40, tmp_path) == [{"type": "header", "raw": b"raw"}] and calls == [("sys", "u", "b" * 40, tmp_path)]


# --- F3: nothing on an answer path reads a keyed field (the content id is sound only while that holds) -----------------------------------------------------
CPG = Path(__file__).resolve().parents[1]
ANSWER_PATH = ("ccg/engine", "ccg/front", "ccg/store", "clients")
#: The writers' own modules (``ccg/linker``, ``ccg/valuehash.py``, ``ccg/atrest.py``, ``ccg/bundle.py``, ``ccg/build.py``, ``ccg/refresh.py``, ``extractors/``) are not on
#: an answer path and are not scanned. The one allowed mention on a scanned path is the loader-side column the assembler writes (``ccg/store`` has none today).
ALLOWED: dict[str, set[str]] = {}


def keyed_mentions(root: Path) -> dict:
    import re
    names = re.compile(r"\b(valueHash|defaultHash|bindKeyCheck|bindParts|hostHmac|pathHmacs)\b")
    hits = {}
    for top in ANSWER_PATH:
        for f in sorted((root / top).rglob("*")) if (root / top).is_dir() else ():
            if f.is_file() and f.suffix in (".py", ".cypher", ".yaml", ".json") and "__pycache__" not in f.parts:
                found = set(names.findall(f.read_text(encoding="utf-8", errors="replace"))) - ALLOWED.get(f.relative_to(root).as_posix(), set())
                if found:
                    hits[f.relative_to(root).as_posix()] = sorted(found)
    return hits


def test_no_answer_path_module_names_a_keyed_bundle_field():
    assert keyed_mentions(CPG) == {}, "a keyed field is read on an answer path: the content id (gate.content_snapshot_id) would no longer be sound"


def test_the_static_guard_reddens_on_a_planted_mention_in_each_scanned_tree(tmp_path):
    for top in ANSWER_PATH:
        d = tmp_path / top
        d.mkdir(parents=True)
        (d / "x.py").write_text("q = 'RETURN e.valueHash'\n")
    assert keyed_mentions(tmp_path) == {f"{top}/x.py": ["valueHash"] for top in ANSWER_PATH}


def test_the_guard_covers_every_answer_path_tree_and_every_keyed_name(tmp_path):
    assert ANSWER_PATH == ("ccg/engine", "ccg/front", "ccg/store", "clients")                  # written out here, not read from the guard
    d = tmp_path / "clients"
    d.mkdir()
    for name in ("valueHash", "defaultHash", "bindKeyCheck", "bindParts", "hostHmac", "pathHmacs"):
        (d / "x.py").write_text(f"q = {name!r}\n")
        assert keyed_mentions(tmp_path) == {"clients/x.py": [name]}
