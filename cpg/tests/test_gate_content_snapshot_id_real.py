"""``gate.store_content_id`` over bundles a REAL refresh wrote (``atrest.write_bundle`` -> ``atrest.read_bundle`` -> ``bundle.loads``), not injected ones.

A failed self-check after a 16-20 minute gate build strands the store, so the read path (work-root layout, build token, stale-kept and not-mapped units) is
proven here on the real writers, in the style of ``tests/test_refresh_scope.py``: one fresh, one stale-kept and one not-mapped unit, under two different system
keys, over the same repository commits."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "eval" / "tools"))
from test_refresh import SYSTEM, World, failing, make_target, uid  # noqa: E402
from test_refresh_scope import rows  # noqa: E402,F401  (the fixture: the rows each refresh gave the loader)

import gate as G  # noqa: E402
from ccg import atrest  # noqa: E402
from ccg import refresh as RF  # noqa: E402

STALE = uid("conf/app.yaml", "config", "r1")          # a config unit: its bundle carries keyed fields
GONE = uid("bin/run.sh", "shell", "r1")


class RowsStore:
    """What ``StoreSnapshot`` reads off the loaded graph, taken from the rows the loader was given (the same values the ``:Snapshot`` and ``:Unit`` nodes hold)."""

    def __init__(self, r):
        self.r = r

    def snapshot_inputs(self):
        s = self.r.snapshot
        return {k: s[k] for k in ("snapshotId", "manifestHash", "packSetHash", "codeHash")}

    def unit_rows(self):
        return sorted([[u["unitId"], u["buildStatus"], u.get("builtCommit"), bool(u["flowAvailable"]), u.get("failReason") or ""] for u in self.r.nodes["Unit"]],
                      key=lambda x: x[0])


def build_under_a_key(w, tag, rows):
    """A work root and keys root of their own over the world's repositories: ``GONE`` never builds (not-mapped), ``STALE`` builds once then fails (stale-kept)."""
    work, keys = w.work.parent / f"work-{tag}", w.work.parent / f"keys-{tag}"
    work.mkdir(mode=0o700)
    kw = dict(work_root=work, keys_root=keys)
    RF.refresh(SYSTEM, **w.kw(make_target()[0], build=failing((GONE, "extract:timeout")), **kw))
    RF.refresh(SYSTEM, **w.kw(make_target()[0], build=failing((GONE, "extract:timeout"), (STALE, "extract:timeout")), **kw))
    return work, rows.rows


@pytest.fixture
def two_builds(tmp_path, rows):
    w = World(tmp_path, repos=("r1",))
    return [build_under_a_key(w, t, rows) for t in ("one", "two")]


def test_the_fixture_builds_a_fresh_a_stale_kept_and_a_not_mapped_unit_under_two_different_keys(two_builds):
    for _, r in two_builds:
        status = {u["unitId"]: u["buildStatus"] for u in r.nodes["Unit"]}
        assert status[STALE] == "stale-kept" and status[GONE] == "not-mapped" and "fresh" in status.values()
    assert two_builds[0][1].snapshot["snapshotId"] != two_builds[1][1].snapshot["snapshotId"]          # the keyed fields reached the stamped id


def test_the_store_reader_reproduces_the_stamped_id_from_real_bundles_and_agrees_across_two_keys(two_builds):
    (work1, r1), (work2, r2) = two_builds
    a = G.store_content_id(RowsStore(r1), SYSTEM, work1)               # the self-check inside it passes on real at-rest bundles
    b = G.store_content_id(RowsStore(r2), SYSTEM, work2)
    assert a == b and a != r1.snapshot["snapshotId"]


def test_a_stale_kept_unit_is_read_from_its_last_good_bundle_not_skipped(two_builds, monkeypatch):
    (work1, r1), _ = two_builds
    read = []
    real = G._stored_bundle
    monkeypatch.setattr(G, "_stored_bundle", lambda system, unit, build, root: read.append(unit) or real(system, unit, build, root))
    G.store_content_id(RowsStore(r1), SYSTEM, work1)
    assert STALE in read and GONE not in read


def test_a_missing_bundle_is_a_value_free_harness_error_with_the_stranded_store_remedy(two_builds):
    (work1, r1), _ = two_builds
    gone = [p for p in atrest.bundle_dir(SYSTEM, work1).iterdir() if p.name.startswith(STALE + ".") and p.name.endswith(atrest.BUNDLE_SUFFIX)]
    assert gone
    for p in gone:
        p.unlink()
    with pytest.raises(G.HarnessError) as e:
        G.store_content_id(RowsStore(r1), SYSTEM, work1)
    text = str(e.value)
    assert G.STORE_REMEDY in text and str(work1) not in text and STALE not in text and not isinstance(e.value, atrest.AtRestError)


def test_an_absent_work_dir_is_the_same_harness_error(two_builds, tmp_path):
    (_, r1), _ = two_builds
    with pytest.raises(G.HarnessError, match="missing or unreadable"):
        G.store_content_id(RowsStore(r1), SYSTEM, tmp_path / "no-such-work")


# --- N1: the keyed shapes a real writer emits (bindParts, bindKeyCheck, defaultHash), not only valueHash ---------------------------------------------------
URL_FILES = {"conf/client.json": '{"url": "http://localhost:8000/mcp/v1"}\n',
             "compose.yaml": "services:\n  web:\n    image: x\n    ports: ['8000:8000']\n    command: ['sh','-c','run']\n"
                             "    environment:\n      - API=http://localhost:8000/api\n      - PORT=${PORT:-8000}\n"}


@pytest.fixture
def url_builds(tmp_path, rows):
    w = World(tmp_path, repos=("r1",), files=URL_FILES)
    out = []
    for tag in ("one", "two"):
        work, keys = tmp_path / f"work-{tag}", tmp_path / f"keys-{tag}"
        work.mkdir(mode=0o700)
        RF.refresh(SYSTEM, **w.kw(make_target()[0], work_root=work, keys_root=keys))
        out.append((work, rows.rows))
    return out


def stored_records(work):
    from ccg import bundle
    out = []
    for p in sorted(atrest.bundle_dir(SYSTEM, work).iterdir()):
        if p.name.endswith(atrest.BUNDLE_SUFFIX):
            out.append(bundle.loads(atrest.read_bundle(SYSTEM, p.name.split(".")[0], p.name.split(".")[1], work)))
    return out


def test_the_real_url_bundles_carry_every_keyed_field_and_their_content_ids_agree_across_two_keys(url_builds):
    (work1, r1), (work2, r2) = url_builds
    recs = [r for b in stored_records(work1) for r in b]
    assert any(r["type"] == "configEntry" and r.get("bindParts", {}).get("pathHmacs") for r in recs)           # the fixture is not vacuous: a keyed host and path
    assert any(r["type"] == "launchFact" and "defaultHash" in r for r in recs) and any("bindKeyCheck" in r for r in recs if r["type"] == "header")
    assert r1.snapshot["snapshotId"] != r2.snapshot["snapshotId"]
    assert G.store_content_id(RowsStore(r1), SYSTEM, work1) == G.store_content_id(RowsStore(r2), SYSTEM, work2)


def test_on_real_bundles_a_changed_port_or_path_depth_moves_the_content_hash_and_other_keys_do_not(url_builds):
    import copy
    (work1, _), (work2, _) = url_builds
    (b1,) = [b for b in stored_records(work1) if any(r["type"] == "configEntry" and "bindParts" in r for r in b)]
    (b2,) = [b for b in stored_records(work2) if any(r["type"] == "configEntry" and "bindParts" in r for r in b) and b[0]["unitId"] == b1[0]["unitId"]]
    base = G.content_bundle_hash(b1)
    assert G.content_bundle_hash(b2) == base                                                            # another key, the same content

    def edited(edit):
        c = copy.deepcopy(b1)
        edit(next(r for r in c if r["type"] == "configEntry" and "bindParts" in r))
        return G.content_bundle_hash(c)

    assert edited(lambda e: e["bindParts"].update(port=e["bindParts"].get("port", 0) + 1)) != base
    assert edited(lambda e: e["bindParts"].update(pathHmacs=e["bindParts"]["pathHmacs"][:-1])) != base       # one segment fewer
