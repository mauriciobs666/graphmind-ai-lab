"""P1-34a: the `p1` flow set of run_fitness.py (flowdraw-p1, --set p1) and s28_metrics.py (--flow-set p1, --record, --join).

Synthetic frames and exclusion files in tmp trees only; no custodian data, no Joern. A p1 run must route EVERY path to a `-p1` name (a
recording filesystem proves no P0 file is opened), the draw must never write/print/log an excluded identity or the unfiltered order, and the
record check must refuse on any pinned-sha mismatch.  Mutants run on COPIES (CCG_FF_DIR, like test_frontend_fitness.py).
"""
import builtins
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REAL = Path(__file__).resolve().parents[1] / "spikes" / "frontend-fitness"
FF = Path(os.environ.get("CCG_FF_DIR") or REAL)
sys.path.insert(0, str(FF))
import run_fitness as rf  # noqa: E402
import s28_core as C  # noqa: E402
import s28_metrics as sm  # noqa: E402
import test_s28_metrics as T  # noqa: E402  (planted python facts/source, shared with the S28 suite)

H = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731
COMMIT = "abc123"
SEED1 = f"ccg-p1a-v1-flow|python|{COMMIT}"


# ------------------------------------------------------------------------------------------------ synthetic corpus
def _t1(name, k):  # a tier-1 method: 7 lines, 1 param, a branch and a call
    return f"def {name}_{k}(x):\n    if x:\n        g(x)\n    y = h(x)\n    z = y\n    return z\n\n"


def _t2(name, k):  # a tier-2 method: 30 lines
    return f"def {name}_{k}(x):\n    if x:\n        g(x)\n" + "    y = h(x)\n" * 26 + "    return y\n\n"


def make_corpus(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    for i in range(14):
        (root / f"f{i}.py").write_text("".join(_t1(f"m{i}", k) for k in range(5)))
    for i in range(6):
        (root / f"t{i}.py").write_text("".join(_t2(f"w{i}", k) for k in range(5)))
    # a file whose only method is the (planted) held-out anchor's: its name must never surface anywhere
    (root / "zz_held.py").write_text(_t1("zz_secret", 0))
    # a nested def: an anchor on the inner def's line sits inside the enclosing method too
    (root / "nest.py").write_text("def outer(x):\n    def inner(y):\n        if y:\n            g(y)\n        return h(y)\n    if x:\n        g(x)\n    return inner(x)\n")


@pytest.fixture
def pop(tmp_path):
    make_corpus(tmp_path / "src")
    return rf.flow_population("python", tmp_path / "src", {})


def oracle_order(pop_, seed):
    return [m for _, m in sorted(((H(seed + "|" + m["siteKey"]), m) for m in pop_), key=lambda t: (t[0], t[1]["siteKey"]))]


def span_has(m, path, line):
    return m["path"] == path and m["lineStart"] <= line <= m["lineEnd"]


# ------------------------------------------------------------------------------------------------ seeds and paths
def test_p1_seeds_are_the_registered_strings_and_p0_is_unchanged():
    assert rf.seed_for("flow", "python", COMMIT, "p1") == SEED1
    assert rf.seed_for("spur", "python", COMMIT, "p1") == f"ccg-p1a-v1-spur|python|{COMMIT}"
    assert rf.seed_for("flow", "python", COMMIT) == f"ccg-p0a-v1-flow|python|{COMMIT}"
    assert rf.seed_for("spur", "python", COMMIT) == f"ccg-p0a-v1-spur|python|{COMMIT}"


def test_every_p1_flow_path_is_distinct_from_p0_and_p0_names_are_unchanged():
    p0 = [rf.flow_path("python", "p0"), rf.flow_tier2_path("python", "p0"), rf.flow_brief_path("python", "p0"), rf.flow_label_path("python", "p0")]
    p1 = [rf.flow_path("python", "p1"), rf.flow_tier2_path("python", "p1"), rf.flow_brief_path("python", "p1"), rf.flow_label_path("python", "p1")]
    assert [p.name for p in p0] == ["python-flow.jsonl", "python-flow-tier2.jsonl", "python-flow.brief.md", "python-flow.jsonl"]
    assert [p.name for p in p1] == ["python-flow-p1.jsonl", "python-flow-p1-tier2.jsonl", "python-flow-p1.brief.md", "python-flow-p1.jsonl"]
    assert not set(p0) & set(p1)
    assert rf.flow_path("python") == p0[0]                       # the default set is p0


def test_metrics_paths_p1_are_all_distinct_from_p0_and_p0_are_the_historical_names():
    a, b = sm.flow_paths("python", "p0"), sm.flow_paths("python", "p1", "fixed-new")
    assert a["results"].name == "projection-python.json" and a["truth"].name == "python.flow-final.jsonl" and a["results"].parent.name == "results"
    assert a["spurious"].name == "python-flow-spurious.jsonl" and a["spur_prefix"] == "ccg-p0a-v1-spur|"
    for k in ("sample", "truth", "audit", "results", "answers", "misses", "spurious"):
        assert a[k] != b[k] and "p1" in str(b[k].relative_to(sm.HERE)), k
    assert b["results"].parent.relative_to(sm.HERE).as_posix() == "results/p1/fixed-new"
    assert {b[k].name for k in ("sample", "truth", "audit", "results", "answers", "misses", "spurious")} == {
        "python-flow-p1.jsonl", "python.flow-p1-final.jsonl", "python-flow-p1-audit.jsonl", "projection-python-p1.json",
        "projection-python-p1.answers.jsonl", "python-flow-p1-misses.jsonl", "python-flow-p1-spurious.jsonl"}
    assert b["spur_prefix"] == "ccg-p1a-v1-spur|"


def test_each_cell_has_its_own_directory_and_only_the_decision_cell_has_adjudication_files():
    cells = ("fixed-new", "fixed-p0", "shipped-new", "shipped-p0")
    ps = {c: sm.flow_paths("python", "p1", c) for c in cells}
    assert len({p["results"].parent for p in ps.values()}) == 4 and len({p["answers"] for p in ps.values()}) == 4
    assert [c for c in cells if ps[c]["misses"] is not None] == ["fixed-new"] and [c for c in cells if ps[c]["spurious"] is not None] == ["fixed-new"]


def test_the_p0_flowdraw_refuses_under_the_p1_set(monkeypatch):
    monkeypatch.setattr(rf, "FLOW_SET", "p1")
    with pytest.raises(SystemExit, match="flowdraw-p1"):
        rf.cmd_flowdraw("python")


# ------------------------------------------------------------------------------------------------ the draw (function level)
def test_no_exclusion_p0_seed_reproduces_the_p0_flowdraw_byte_for_byte(tmp_path, monkeypatch, pop):
    ff = tmp_path / "ff"
    monkeypatch.setattr(rf, "HERE", ff)
    monkeypatch.setattr(rf, "corpus_cfg", lambda l: {"commit": COMMIT})
    monkeypatch.setattr(rf, "src_dir", lambda l: tmp_path / "src")
    rf.cmd_flowdraw("python")
    p0_bytes = (ff / "sample" / "python-flow.jsonl").read_bytes()
    rows, kf = rf.draw_flow_p1(pop, rf.seed_for("flow", "python", COMMIT), 30, set(), [])
    out = tmp_path / "again.jsonl"
    rf.write_jsonl(out, rows)
    assert out.read_bytes() == p0_bytes and kf == 0 and len(rows) == 30


def test_order_is_ascending_sha256_of_seed_and_sitekey_first_n(pop):
    rows, _k = rf.draw_flow_p1(pop, SEED1, 30, set(), [])
    want = oracle_order(pop, SEED1)[:30]
    assert [(r["path"], r["qname"]) for r in rows] == [(m["path"], m["qname"]) for m in want]
    assert [r["ordinal"] for r in rows] == list(range(1, 31))


def test_the_seed_changes_the_draw(pop):
    a, _k = rf.draw_flow_p1(pop, SEED1, 30, set(), [])
    b, _k = rf.draw_flow_p1(pop, f"ccg-p0a-v1-flow|python|{COMMIT}", 30, set(), [])
    assert [r["qname"] for r in a] != [r["qname"] for r in b]


def test_p0_methods_are_removed_from_the_frame(pop):
    first = oracle_order(pop, SEED1)[:3]
    rows, _k = rf.draw_flow_p1(pop, SEED1, 30, {m["siteKey"] for m in first}, [])
    assert not {r["qname"] for r in rows} & {m["qname"] for m in first}
    assert [r["qname"] for r in rows] == [m["qname"] for m in oracle_order(pop, SEED1)[3:33]]


def test_a_planted_exclusion_removes_exactly_the_planted_method_and_the_next_one_moves_up(pop):
    order = oracle_order(pop, SEED1)
    victim = order[4]
    excl = [(victim["path"], victim["lineStart"] + 2)]            # an interior line of the method
    rows, kf = rf.draw_flow_p1(pop, SEED1, 30, set(), excl)
    assert [r["qname"] for r in rows] == [m["qname"] for m in order[:4] + order[5:31]]
    assert kf == 1


def test_exclusion_matches_on_file_and_span_not_on_file_alone(pop):
    order = oracle_order(pop, SEED1)
    victim = order[0]
    same_file_others = [m for m in pop if m["path"] == victim["path"] and m["siteKey"] != victim["siteKey"]]
    assert same_file_others                                        # the fixture has siblings in the file
    rows, kf = rf.draw_flow_p1(pop, SEED1, len(pop), set(), [(victim["path"], victim["lineEnd"])])
    got = {r["qname"] for r in rows}
    assert victim["qname"] not in got and all(m["qname"] in got for m in same_file_others) and kf == 1


def test_exclusion_needs_the_file_a_line_in_another_file_or_a_gap_excludes_nothing(pop):
    v = oracle_order(pop, SEED1)[0]
    other_file = next(m["path"] for m in pop if m["path"] != v["path"])
    gap = v["lineEnd"] + 1                                          # the blank line after the method: in no span of its file
    assert not any(span_has(m, v["path"], gap) for m in pop)       # the gap really is outside every span of the file
    other_hits = [m for m in pop if span_has(m, other_file, v["lineStart"] + 1)]   # same line number, another file: only that file's own span hits
    _, kf = rf.draw_flow_p1(pop, SEED1, 30, set(), [(other_file, v["lineStart"] + 1)])
    assert kf == len(other_hits) and v["siteKey"] not in {m["siteKey"] for m in other_hits}
    rows, kf = rf.draw_flow_p1(pop, SEED1, 30, set(), [(v["path"], gap)])
    assert kf == 0 and [r["qname"] for r in rows] == [m["qname"] for m in oracle_order(pop, SEED1)[:30]]


def test_the_enclosing_method_of_a_nested_anchor_is_excluded_too(pop):
    by = {m["qname"].split("::")[1]: m for m in pop if m["path"] == "nest.py"}
    outer = by["outer"]
    inner_line = outer["lineStart"] + 1                             # `def inner` line, inside outer's span
    rows, kf = rf.draw_flow_p1(pop, SEED1, len(pop), set(), [("nest.py", inner_line)])
    assert "nest.py::outer" not in {r["qname"] for r in rows} and kf >= 1


def test_the_frame_count_is_independent_of_where_the_excluded_methods_sit_in_the_order(pop):
    order = oracle_order(pop, SEED1)
    excl = [(order[7]["path"], order[7]["lineStart"]), (order[60]["path"], order[60]["lineStart"])]
    for n in (5, 15, 30):
        assert rf.draw_flow_p1(pop, SEED1, n, set(), excl)[1] == 2


def test_the_draw_returns_no_prefix_count(pop):
    out = rf.draw_flow_p1(pop, SEED1, 30, set(), [])
    assert len(out) == 2


def test_sample_rows_carry_only_the_p0_fields_no_hash_no_position(pop):
    rows, _k = rf.draw_flow_p1(pop, SEED1, 30, set(), [(pop[0]["path"], pop[0]["lineStart"])])
    assert all(set(r) == {"ordinal", "path", "qname", "lineStart", "lineEnd", "source"} for r in rows)


# ------------------------------------------------------------------------------------------------ the command (tmp tree, recording fs)
class Recorder:
    """Records every path opened (with mode) and every os.replace target while active."""

    def __init__(self, monkeypatch):
        self.opened, self.replaced = [], []
        real_open, real_replace = builtins.open, os.replace

        def spy_open(f, mode="r", *a, **k):
            if isinstance(f, (str, os.PathLike)):
                self.opened.append((str(f), mode))
            return real_open(f, mode, *a, **k)

        def spy_replace(a, b, *x, **k):
            self.replaced.append(str(b))
            return real_replace(a, b, *x, **k)

        monkeypatch.setattr(builtins, "open", spy_open)
        monkeypatch.setattr(os, "replace", spy_replace)
        import pathlib
        real_pw = pathlib.Path.write_text
        real_pb = pathlib.Path.write_bytes
        monkeypatch.setattr(pathlib.Path, "write_text", lambda s, *a, **k: (self.opened.append((str(s), "w")), real_pw(s, *a, **k))[1])
        monkeypatch.setattr(pathlib.Path, "write_bytes", lambda s, *a, **k: (self.opened.append((str(s), "wb")), real_pb(s, *a, **k))[1])
        real_ro = pathlib.Path.read_text
        monkeypatch.setattr(pathlib.Path, "read_text", lambda s, *a, **k: (self.opened.append((str(s), "r")), real_ro(s, *a, **k))[1])

    def writes(self):
        return [p for p, m in self.opened if set(m) & set("wax+")] + self.replaced

    def reads(self):
        return [p for p, m in self.opened if not set(m) & set("wax+")]


SENTINELS = {"sample/python-flow.jsonl": "P0-SAMPLE\n", "sample/python-flow-tier2.jsonl": "P0-T2\n", "sample/python-flow.brief.md": "P0-BRIEF\n",
             "labels/python-flow.jsonl": "P0-LABELS\n"}


@pytest.fixture
def env(tmp_path, monkeypatch, pop):
    ff = tmp_path / "ff"
    for rel, txt in SENTINELS.items():
        (ff / rel).parent.mkdir(parents=True, exist_ok=True)
        (ff / rel).write_text(txt)
    # the P0 sample = three real methods of the frame (so the P0-removal check is satisfied)
    order = oracle_order(pop, f"ccg-p0a-v1-flow|python|{COMMIT}")[:3]
    rf.write_jsonl(ff / "sample" / "python-flow.jsonl", [{"ordinal": i, "path": m["path"], "qname": m["qname"], "lineStart": m["lineStart"],
                                                         "lineEnd": m["lineEnd"], "source": m["source"]} for i, m in enumerate(order, 1)])
    SENTINELS["sample/python-flow.jsonl"] = (ff / "sample" / "python-flow.jsonl").read_text()
    monkeypatch.setattr(rf, "HERE", ff)
    monkeypatch.setattr(rf, "corpus_cfg", lambda l: {"commit": COMMIT})
    monkeypatch.setattr(rf, "src_dir", lambda l: tmp_path / "src")
    monkeypatch.setattr(rf, "FLOW_SET", "p0")
    held = [m for m in pop if m["path"] == "zz_held.py"][0]
    priv = tmp_path / "private" / "excl.jsonl"
    priv.parent.mkdir()
    # anchors: the zz_held method (its name must never surface) and one more in the unfiltered top 30
    top = oracle_order([m for m in pop if m["siteKey"] not in {f"{o['path']}:{o['lineStart']}:{o['qname']}" for o in order}], SEED1)[2]
    priv.write_text(json.dumps({"path": "zz_held.py", "defLine": held["lineStart"]}) + "\n" + json.dumps({"path": top["path"], "defLine": top["lineStart"] + 1}) + "\n")
    pop2 = rf.flow_population("python", tmp_path / "src", {}, 26, 60)
    expect = f"{len(pop)},{len(pop) - len(order)},{len(pop2)}"                               # the synthetic frame's own counts
    return {"ff": ff, "priv": priv, "counts": tmp_path / "out" / "counts.json", "tier2": tmp_path / "cust" / "tier2.jsonl", "expect": expect, "top": top, "p0": order}


def tree(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_flowdraw_p1_writes_only_p1_files_and_counts_and_never_opens_a_p0_file_for_write(env, tmp_path, monkeypatch):
    before = tree(tmp_path)
    rec = Recorder(monkeypatch)
    assert rf.main(["flowdraw-p1", "python", "--exclude", str(env["priv"]), "--counts", str(env["counts"]), "--tier2-out", str(env["tier2"]), "--expect-counts", env["expect"]]) == 0
    after = tree(tmp_path)
    new = sorted(set(after) - set(before))
    assert new == ["cust/tier2.jsonl", "ff/sample/python-flow-p1.jsonl", "out/counts.json"]
    assert {k: v for k, v in before.items() if after[k] != v} == {}              # nothing pre-existing changed (P0 files, the private file)
    for w in rec.writes():
        assert "-p1" in Path(w).name or Path(w).name.startswith(("counts.json", "tier2.jsonl")), w
        assert Path(w).name not in {"python-flow.jsonl", "python-flow-tier2.jsonl", "python-flow.brief.md"}
    assert not [r for r in rec.reads() if Path(r).name in {"python-flow-tier2.jsonl", "python-flow.brief.md"}]


def test_flowdraw_p1_sample_files_are_the_filtered_draw_and_exclude_the_planted_anchors(env, pop, tmp_path):
    rf.main(["flowdraw-p1", "python", "--exclude", str(env["priv"]), "--counts", str(env["counts"]), "--tier2-out", str(env["tier2"]), "--expect-counts", env["expect"]])
    s1 = rf.read_jsonl(env["ff"] / "sample" / "python-flow-p1.jsonl")
    s2 = rf.read_jsonl(env["tier2"])
    assert len(s1) == 30 and len(s2) == 15
    drop = {f"{o['path']}:{o['lineStart']}:{o['qname']}" for o in env["p0"]}
    names1 = [r["qname"] for r in s1]
    assert not any("zz_secret" in n for n in names1 + [r["qname"] for r in s2])
    assert env["top"]["qname"] not in names1
    assert not {r["qname"] for r in s1} & {o["qname"] for o in env["p0"]}
    want = [m for m in oracle_order(pop, SEED1) if m["siteKey"] not in drop and m["path"] != "zz_held.py" and m["siteKey"] != env["top"]["siteKey"]]
    assert names1 == [m["qname"] for m in want[:30]]
    pop2 = rf.flow_population("python", tmp_path / "src", {}, 26, 60)
    assert [r["qname"] for r in s2] == [m["qname"] for m in oracle_order(pop2, SEED1)[:15]]    # tier-2 uses the same seed and rule


def test_counts_file_has_exactly_the_two_frame_counts(env, capsys):
    rf.main(["flowdraw-p1", "python", "--exclude", str(env["priv"]), "--counts", str(env["counts"]), "--tier2-out", str(env["tier2"]), "--expect-counts", env["expect"]])
    c = json.loads(env["counts"].read_text())
    assert set(c) == {"excludedTier1Frame", "excludedTier2Frame"}      # exactly two integers: no prefix counts, no digests
    assert all(type(c[k]) is int for k in ("excludedTier1Frame", "excludedTier2Frame"))
    assert c["excludedTier1Frame"] == 2 and c["excludedTier2Frame"] == 0
    out = json.loads(capsys.readouterr().out)                                        # stdout = the same two integers + the grammar versions, nothing else
    assert {k: out[k] for k in c} == c and set(out) == set(c) | {"treeSitter", "grammarPython"}
    assert re.fullmatch(r"\d+\.\d+\.\d+", out["treeSitter"]) and re.fullmatch(r"\d+\.\d+\.\d+", out["grammarPython"])


def test_flowdraw_p1_leaks_neither_an_excluded_identity_nor_the_unfiltered_order(env, tmp_path, capsys, caplog):
    import logging
    caplog.set_level(logging.DEBUG)
    rf.main(["flowdraw-p1", "python", "--exclude", str(env["priv"]), "--counts", str(env["counts"]), "--tier2-out", str(env["tier2"]), "--expect-counts", env["expect"]])
    cap = capsys.readouterr()
    blobs = {"stdout": cap.out, "stderr": cap.err, "log": caplog.text}
    secret = ["zz_held", "zz_secret", env["top"]["qname"], env["top"]["path"] + ":" + str(env["top"]["lineStart"])]
    for p, data in tree(tmp_path).items():
        if p.startswith(("private/", "src/")):
            continue                       # the exclusion file itself and the synthetic corpus (which holds the anchor's method by design)
        if p.startswith("ff/sample/python-flow-p1"):
            continue                       # the sample lists non-anchor methods; their absence is checked in the sample test
        blobs[p] = data.decode("utf-8", "replace")
    for where, text in blobs.items():
        for s in secret:
            assert s not in text, (where, s)
    # no 64-hex (a hash of the unfiltered order) outside the two digests
    for where, text in blobs.items():
        assert not re.findall(r"\b[0-9a-f]{32,}\b", text), where
    assert env["priv"].exists()                                                     # kept where it is, untouched
    assert not [p for p in tmp_path.rglob("*") if "excl" in p.name and p != env["priv"]]


def test_a_malformed_exclusion_line_is_reported_by_number_without_echoing_it(env):
    env["priv"].write_text('{"path": "zz_held.py", "defLine": 3}\nTOPSECRET-not-json\n')
    with pytest.raises(SystemExit) as ei:
        rf.main(["flowdraw-p1", "python", "--exclude", str(env["priv"]), "--counts", str(env["counts"]), "--tier2-out", str(env["tier2"]), "--expect-counts", env["expect"]])
    assert "line 2" in str(ei.value) and "TOPSECRET" not in str(ei.value) and "zz_held" not in str(ei.value)
    env["priv"].write_text('{"path": "zz_held.py", "defLine": "3"}\n')
    with pytest.raises(SystemExit) as ei:
        rf.main(["flowdraw-p1", "python", "--exclude", str(env["priv"]), "--counts", str(env["counts"]), "--tier2-out", str(env["tier2"]), "--expect-counts", env["expect"]])
    assert "zz_held" not in str(ei.value)


def _draw_argv(env):
    return ["flowdraw-p1", "python", "--exclude", str(env["priv"]), "--counts", str(env["counts"]), "--tier2-out", str(env["tier2"]), "--expect-counts", env["expect"]]


HELD_LINES = 7                                                    # zz_held.py: 6 code lines + 1 blank (the file's last line)


@pytest.mark.parametrize("bad", [
    {"path": "SECRETPATH/no/such/file.py", "defLine": 1},          # a file that is not in the corpus
    {"path": "falkor-chat/server/zz_held.py", "defLine": 1},        # the repo-relative form: it matches nothing under the corpus root
    {"path": "./zz_held.py", "defLine": 1},                         # exists once resolved, but is not the strict form
    {"path": "/zz_held.py", "defLine": 1},
    {"path": "sub/../zz_held.py", "defLine": 1},
    {"path": "zz_held.py\\", "defLine": 1},
    {"path": "", "defLine": 1},
    {"path": "held.py", "defLine": 1},                              # a suffix of zz_held.py: matching by suffix is not matching
    {"path": "zz_held", "defLine": 1},
    {"path": "zz_held.py", "defLine": 0},
    {"path": "zz_held.py", "defLine": -3},
    {"path": "zz_held.py", "defLine": HELD_LINES + 1},              # beyond EOF
])
def test_an_exclusion_entry_that_is_not_a_corpus_file_and_line_aborts_by_line_number_only_and_writes_nothing(env, tmp_path, bad):
    good = json.dumps({"path": "zz_held.py", "defLine": 3})
    env["priv"].write_text(good + "\n" + json.dumps(bad) + "\n")
    before = tree(tmp_path)
    with pytest.raises(SystemExit) as ei:
        rf.main(_draw_argv(env))
    msg = str(ei.value)
    assert "line 2" in msg and "line 1" not in msg
    assert not any(t in msg for t in ("SECRETPATH", "zz_held", "held.py", "server", "sub", "999"))
    assert ei.value.__cause__ is None and tree(tmp_path) == before and not env["counts"].exists()


def test_an_exclusion_entry_resolving_outside_the_corpus_through_a_symlink_is_refused(env, tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n" * 5)
    try:
        (tmp_path / "src" / "link.py").symlink_to(outside)
    except OSError:
        pytest.skip("no symlinks here")
    env["priv"].write_text(json.dumps({"path": "zz_held.py", "defLine": 3}) + "\n" + json.dumps({"path": "link.py", "defLine": 2}) + "\n")
    with pytest.raises(SystemExit, match="line 2"):
        rf.main(_draw_argv(env))


def test_the_first_and_the_last_line_of_a_corpus_file_are_valid_exclusion_lines(env):
    env["priv"].write_text(json.dumps({"path": "zz_held.py", "defLine": 1}) + "\n" + json.dumps({"path": "zz_held.py", "defLine": HELD_LINES}) + "\n")
    assert rf.main(_draw_argv(env)) == 0


def test_a_file_without_a_trailing_newline_counts_its_last_line(env, tmp_path):
    (tmp_path / "src" / "nonl.py").write_text("x = 1\ny = 2")           # 2 lines, no final newline
    pop2 = rf.flow_population("python", tmp_path / "src", {})
    pop3 = rf.flow_population("python", tmp_path / "src", {}, 26, 60)
    assert (len(pop2), len(pop3)) == tuple(int(x) for x in env["expect"].split(","))[::2]        # the extra file holds no method: counts unchanged
    env["priv"].write_text(json.dumps({"path": "nonl.py", "defLine": 2}) + "\n" + json.dumps({"path": "zz_held.py", "defLine": 3}) + "\n")
    assert rf.main(_draw_argv(env)) == 0
    env["priv"].write_text(json.dumps({"path": "nonl.py", "defLine": 3}) + "\n" + json.dumps({"path": "zz_held.py", "defLine": 3}) + "\n")
    with pytest.raises(SystemExit, match="line 1"):
        rf.main(_draw_argv(env))


def test_valid_entries_that_lie_in_no_frame_method_draw_and_report_zero_zero(env, tmp_path):
    env["priv"].write_text(json.dumps({"path": "zz_held.py", "defLine": HELD_LINES}) + "\n")      # a valid line (the blank last one) in no method's span
    assert rf.main(_draw_argv(env)) == 0
    assert json.loads(env["counts"].read_text()) == {"excludedTier1Frame": 0, "excludedTier2Frame": 0}
    assert len(rf.read_jsonl(env["ff"] / "sample" / "python-flow-p1.jsonl")) == 30 and len(rf.read_jsonl(env["tier2"])) == 15


def test_a_non_utf8_exclusion_file_is_a_value_free_error(env, tmp_path):
    env["priv"].write_bytes(b'{"path": "zz_held.py", "defLine": 3}\n\xff\xfeSECRET\n')
    before = tree(tmp_path)
    with pytest.raises(SystemExit) as ei:
        rf.main(_draw_argv(env))
    msg = str(ei.value)
    assert not any(t in msg.lower() for t in ("byte", "position", "0x", "xff", "codec", "secret", "zz_held")) and ei.value.__cause__ is None and ei.value.__suppress_context__
    assert tree(tmp_path) == before


def test_a_deeply_nested_exclusion_line_is_a_line_number_error(env):
    env["priv"].write_text('{"path": "zz_held.py", "defLine": 3}\n' + "[" * 200000 + "\n")
    with pytest.raises(SystemExit, match="line 2") as ei:
        rf.main(_draw_argv(env))
    assert ei.value.__cause__ is None


def test_an_unwritable_tier2_location_leaves_nothing_behind_not_even_the_tier1_sample(env, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, so nothing can be created under it")
    env["tier2"] = blocker / "sub" / "tier2.jsonl"
    before = tree(tmp_path)
    with pytest.raises(SystemExit, match="nothing written") as ei:
        rf.main(_draw_argv(env))
    assert "blocker" not in str(ei.value)
    assert tree(tmp_path) == before and not (env["ff"] / "sample" / "python-flow-p1.jsonl").exists() and not env["counts"].exists()


def test_an_unwritable_counts_location_after_the_others_were_staged_leaves_no_file_and_no_temp_file(env, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file")
    env["counts"] = blocker / "sub" / "counts.json"
    before = tree(tmp_path)
    with pytest.raises(SystemExit, match="nothing written"):
        rf.main(_draw_argv(env))
    assert tree(tmp_path) == before and not list(tmp_path.rglob("*.tmp"))


def test_the_outputs_are_staged_then_renamed_so_no_output_is_replaced_before_all_are_staged(env, tmp_path, monkeypatch):
    seen = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b, *x, **k: (seen.append([p.name for p in tmp_path.rglob("*.tmp")]), real(a, b, *x, **k))[1])
    rf.main(_draw_argv(env))
    assert seen and all(len(s) == 3 for s in seen[:1])                 # at the first rename all three outputs already sit staged as temp files
    assert not list(tmp_path.rglob("*.tmp"))


def test_flowdraw_p1_fails_closed_without_exclude_or_counts_and_writes_nothing(env, tmp_path, capsys):
    before = tree(tmp_path)
    for argv in (["flowdraw-p1", "python", "--counts", str(env["counts"]), "--tier2-out", str(env["tier2"]), "--expect-counts", env["expect"]], ["flowdraw-p1", "python", "--exclude", str(env["priv"])], ["flowdraw-p1", "python"]):
        with pytest.raises(SystemExit) as ei:
            rf.main(argv)
        assert ei.value.code == 2
    capsys.readouterr()
    assert tree(tmp_path) == before and not (env["ff"] / "sample" / "python-flow-p1.jsonl").exists()
    with pytest.raises(SystemExit) as ei:
        rf.main(["flowdraw-p1", "python", "--exclude", str(env["priv"]), "--counts", str(env["counts"])])      # no --tier2-out
    assert ei.value.code == 2 and tree(tmp_path) == before


@pytest.mark.parametrize("content", ["", "\n\n  \n"])
def test_an_empty_exclusion_file_aborts_and_writes_nothing(env, tmp_path, content):
    env["priv"].write_text(content)
    before = tree(tmp_path)
    with pytest.raises(SystemExit, match="no entries"):
        rf.main(["flowdraw-p1", "python", "--exclude", str(env["priv"]), "--counts", str(env["counts"]), "--tier2-out", str(env["tier2"]), "--expect-counts", env["expect"]])
    assert tree(tmp_path) == before


def test_tier2_out_inside_the_repository_is_refused(env, tmp_path, monkeypatch):
    repo = tmp_path / "fakerepo"                                   # hermetic: a failure of the guard must not write into the real repository
    monkeypatch.setattr(rf, "REPO", repo)
    for bad in (repo / "tier2.jsonl", repo / "cpg" / "sub" / "t2.jsonl", repo / "cpg" / "sample" / "python-flow-tier2.jsonl"):
        with pytest.raises(SystemExit, match="outside the repository"):
            rf.main(["flowdraw-p1", "python", "--exclude", str(env["priv"]), "--counts", str(env["counts"]), "--tier2-out", str(bad)])
    assert not env["counts"].exists() and not (env["ff"] / "sample" / "python-flow-p1.jsonl").exists()


@pytest.mark.parametrize("counts", [(None, 0), (None, 1), (None, 2)])
def test_a_planted_frame_count_discrepancy_aborts_before_the_exclusion_with_no_file_written(env, tmp_path, counts):
    good = [int(x) for x in env["expect"].split(",")]
    good[counts[1]] += 1
    before = tree(tmp_path)
    with pytest.raises(SystemExit, match="frame counts") as ei:
        rf.main(["flowdraw-p1", "python", "--exclude", str(env["priv"]), "--counts", str(env["counts"]), "--tier2-out", str(env["tier2"]), "--expect-counts", ",".join(map(str, good))])
    assert "zz_held" not in str(ei.value) and tree(tmp_path) == before


REAL_CORPUS = REAL.parents[1] / ".ccg-work" / "corpora" / "python" / "src"


@pytest.mark.skipif(not REAL_CORPUS.exists(), reason="the fetched python corpus is not on this machine")
def test_real_corpus_with_a_test_seed_and_a_dummy_exclusion_has_the_registered_frame_counts(tmp_path, monkeypatch):
    ff = tmp_path / "ff"
    (ff / "sample").mkdir(parents=True)
    shutil.copy(REAL / "sample" / "python-flow.jsonl", ff / "sample" / "python-flow.jsonl")       # the 30 public P0 methods
    monkeypatch.setattr(rf, "HERE", ff)
    monkeypatch.setattr(rf, "corpus_cfg", lambda l: {"commit": "TEST-SEED-not-the-registered-one"})   # NEVER the real seed
    monkeypatch.setattr(rf, "src_dir", lambda l: REAL_CORPUS)
    monkeypatch.setattr(rf, "FLOW_SET", "p0")
    pop1 = rf.flow_population("python", REAL_CORPUS, {})
    p0 = {f"{r['path']}:{r['lineStart']}:{r['qname']}" for r in rf.read_jsonl(ff / "sample" / "python-flow.jsonl")}
    victim = next(m for m in pop1 if m["siteKey"] not in p0)               # a dummy anchor: the definition line of one real frame method
    dummy = tmp_path / "dummy.jsonl"
    dummy.write_text(json.dumps({"path": victim["path"], "defLine": victim["lineStart"]}) + "\n")
    assert rf.seed_for("flow", "python", rf.corpus_cfg("python")["commit"], "p1") != rf.seed_for("flow", "python", rf.P1_REAL_COMMIT, "p1")   # never the registered seed
    assert rf.P1_FRAME_COUNTS == (368, 338, 163)
    assert rf.main(["flowdraw-p1", "python", "--exclude", str(dummy), "--counts", str(tmp_path / "c.json"), "--tier2-out", str(tmp_path / "t2.jsonl")]) == 0
    want = sum(1 for m in pop1 if m["siteKey"] not in p0 and span_has(m, victim["path"], victim["lineStart"]))
    assert json.loads((tmp_path / "c.json").read_text()) == {"excludedTier1Frame": want, "excludedTier2Frame": 0} and want >= 1
    assert len(rf.read_jsonl(ff / "sample" / "python-flow-p1.jsonl")) == 30 and len(rf.read_jsonl(tmp_path / "t2.jsonl")) == 15


def test_an_unreadable_exclusion_path_is_not_echoed(env, tmp_path):
    missing = tmp_path / "private" / "SECRET-DIR-xyz" / "nope.jsonl"
    with pytest.raises(SystemExit) as ei:
        rf.main(["flowdraw-p1", "python", "--exclude", str(missing), "--counts", str(env["counts"]), "--tier2-out", str(env["tier2"]), "--expect-counts", env["expect"]])
    assert "SECRET-DIR-xyz" not in str(ei.value) and ei.value.__cause__ is None and ei.value.__suppress_context__
    assert not (env["ff"] / "sample" / "python-flow-p1.jsonl").exists()


def test_the_registered_seed_is_refused_on_any_other_corpus_and_a_test_seed_is_not(env, monkeypatch):
    monkeypatch.setattr(rf, "P1_REAL_COMMIT", "TEST-REGISTERED-not-the-real-commit")       # the guard is exercised with a TEST value: no test runs the real seed string
    monkeypatch.setattr(rf, "corpus_cfg", lambda l: {"commit": "TEST-REGISTERED-not-the-real-commit"})
    assert "0caf85e3" not in rf.seed_for("flow", "python", rf.corpus_cfg("python")["commit"], "p1")
    with pytest.raises(SystemExit, match="registered corpus"):
        rf.main(["flowdraw-p1", "python", "--exclude", str(env["priv"]), "--counts", str(env["counts"]), "--tier2-out", str(env["tier2"]), "--expect-counts", env["expect"]])
    assert not (env["ff"] / "sample" / "python-flow-p1.jsonl").exists() and not env["counts"].exists()
    monkeypatch.setattr(rf, "tree_digest", lambda root: rf.P1_REAL_SRC_DIGEST)       # the registered corpus: allowed
    n1, n2, n3 = (int(x) for x in env["expect"].split(","))
    rf.write_jsonl(env["ff"] / "sample" / "python-flow.jsonl", [])
    assert rf.main(["flowdraw-p1", "python", "--exclude", str(env["priv"]), "--counts", str(env["counts"]), "--tier2-out", str(env["tier2"]), "--expect-counts", f"{n1},{n1},{n3}"]) == 0


def test_the_default_frame_counts_apply_only_on_the_registered_corpus(env, tmp_path):
    before = tree(tmp_path)
    with pytest.raises(SystemExit, match="registered corpus only"):
        rf.main(["flowdraw-p1", "python", "--exclude", str(env["priv"]), "--counts", str(env["counts"]), "--tier2-out", str(env["tier2"])])
    assert tree(tmp_path) == before


def test_the_python_tier2_sample_is_15_methods_of_26_to_60_lines_disjoint_from_tier1(env, tmp_path):
    rf.main(["flowdraw-p1", "python", "--exclude", str(env["priv"]), "--counts", str(env["counts"]), "--tier2-out", str(env["tier2"]), "--expect-counts", env["expect"]])
    t1 = rf.read_jsonl(env["ff"] / "sample" / "python-flow-p1.jsonl")
    t2 = rf.read_jsonl(env["tier2"])
    assert not (env["ff"] / "sample" / "python-flow-p1-tier2.jsonl").exists()        # tier-2 goes only to --tier2-out
    assert len(t2) == 15 and all(26 <= r["lineEnd"] - r["lineStart"] + 1 <= 60 for r in t2)
    assert not {(r["path"], r["qname"]) for r in t1} & {(r["path"], r["qname"]) for r in t2}
    assert [r["ordinal"] for r in t2] == list(range(1, 16))


def test_flowdraw_p1_refuses_when_a_p0_method_left_the_population(env):
    rf.write_jsonl(env["ff"] / "sample" / "python-flow.jsonl", [{"ordinal": 1, "path": "gone.py", "qname": "gone.py::x", "lineStart": 1, "lineEnd": 9, "source": ""}])
    with pytest.raises(SystemExit, match="P0 sample method"):
        rf.main(["flowdraw-p1", "python", "--exclude", str(env["priv"]), "--counts", str(env["counts"]), "--tier2-out", str(env["tier2"]), "--expect-counts", env["expect"]])


def test_flowdraw_p1_refuses_to_write_a_non_p1_sample_path(env, monkeypatch):
    monkeypatch.setattr(rf, "flow_path", lambda l, fs=None: env["ff"] / "sample" / "python-flow.jsonl")
    with pytest.raises(SystemExit, match="non-p1"):
        rf.main(["flowdraw-p1", "python", "--exclude", str(env["priv"]), "--counts", str(env["counts"]), "--tier2-out", str(env["tier2"]), "--expect-counts", env["expect"]])
    assert (env["ff"] / "sample" / "python-flow.jsonl").read_text() == SENTINELS["sample/python-flow.jsonl"]


# ------------------------------------------------------------------------------------------------ brief and check under --set p1
def test_flowbrief_and_flowcheck_set_p1_use_only_p1_paths(env, tmp_path, monkeypatch):
    rf.main(["flowdraw-p1", "python", "--exclude", str(env["priv"]), "--counts", str(env["counts"]), "--tier2-out", str(env["tier2"]), "--expect-counts", env["expect"]])
    before = tree(env["ff"])
    rec = Recorder(monkeypatch)
    assert rf.main(["flowbrief", "python", "--set", "p1"]) == 0
    brief = env["ff"] / "sample" / "python-flow-p1.brief.md"
    assert brief.exists() and "python-flow-p1.jsonl" in brief.read_text() and "labels/python-flow-p1.jsonl" in brief.read_text()
    assert (env["ff"] / "sample" / "python-flow.brief.md").read_text() == "P0-BRIEF\n"
    rf.main(["flowcheck", "python", "--set", "p1"])
    changed = {k for k, v in tree(env["ff"]).items() if before.get(k) != v}
    assert changed == {"sample/python-flow-p1.brief.md"}
    assert "python-flow.jsonl" not in {Path(p).name for p in rec.reads() if "/ff/" in p}
    assert (env["ff"] / "labels" / "python-flow.jsonl").read_text() == "P0-LABELS\n"


def test_default_set_is_p0_for_brief_and_check(env):
    assert rf.main(["flowbrief", "python"]) == 0
    assert (env["ff"] / "sample" / "python-flow.brief.md").read_text() != "P0-BRIEF\n"
    assert not (env["ff"] / "sample" / "python-flow-p1.brief.md").exists() and not (env["ff"] / "sample" / "python-flow-p1.jsonl").exists()


# ------------------------------------------------------------------------------------------------ s28_metrics: p1 runs
def p1_world(tmp_path, monkeypatch, run_extra=None):
    """A tmp ff/ + work tree holding the planted python facts, a P1 sample/truth/audit and P0 sentinels at every P0 name."""
    ff, work, corp = tmp_path / "ff", tmp_path / "work" / "python", tmp_path / "corp"
    for d in ("sample", "truth", "adjudication", "results"):
        (ff / d).mkdir(parents=True, exist_ok=True)
    (ff / "corpora.yaml").write_text(f"corpora:\n  python:\n    commit: {COMMIT}\n")
    (corp / "python" / "src").mkdir(parents=True)
    (corp / "python" / "src" / "a.py").write_text(T.PY_SRC)
    work.mkdir(parents=True)
    facts = work / "facts.jsonl"
    facts.write_text("\n".join(json.dumps(r) for r in T.py_facts()) + "\n")
    ext_sha = hashlib.sha256((REAL.parents[1] / "extractors" / "joern" / "extract.sc").read_bytes()).hexdigest()
    for c in ("off", "on"):
        (work / f"{c}.cpg.bin").write_bytes(f"CPG-{c}".encode())
    cfg = {"rc": 0, "wallSeconds": 1.0, "factsFile": str(facts), "factsMB": 0.1, "peakMemMiB": 1, "startedUtc": "x"}
    mk = lambda sha: {"runs": {"off.extract": dict(cfg, extractSha256=sha, cpgFile=str(work / "off.cpg.bin")),   # noqa: E731
                               "on.extract": dict(cfg, extractSha256=sha, cpgFile=str(work / "on.cpg.bin")),
                               "off.content-check": {"contentLines": ["METHOD_COUNT=1"]}, "on.content-check": {"contentLines": ["METHOD_COUNT=1"]}}}
    (work / "s28-run.json").write_text(json.dumps(mk(ext_sha)))                      # the fixed run
    (work / "other-run.json").write_text(json.dumps(mk(None)))                       # the shipped P0 record has no script sha
    (work / "s27-run.json").write_text(json.dumps({"runs": {"off.parse": {"cpgMB": 1.0, "wallSeconds": 1.0}, "on.parse": {"cpgMB": 1.0, "wallSeconds": 1.0}}}))
    samp = [{"ordinal": 1, "path": "a.py", "qname": "a.py::f", "lineStart": 1, "lineEnd": 5}]
    gold = [{"ordinal": 1, "path": "a.py", "pairs": [{"kind": k, "source": s, "sink": t, "confidence": "sure"} for _, k, s, t, *_ in T.PY_PAIRS]}]
    for name, rows in (("sample/python-flow-p1.jsonl", samp), ("truth/python.flow-p1-final.jsonl", gold), ("adjudication/python-flow-p1-audit.jsonl", []),
                       ("sample/python-flow.jsonl", samp), ("truth/python.flow-final.jsonl", gold), ("adjudication/python-flow-audit.jsonl", [])):
        (ff / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    for name in ("results/projection-python.json", "results/projection-python.answers.jsonl", "adjudication/python-flow-misses.jsonl",
                 "adjudication/python-flow-spurious.jsonl"):
        (ff / name).write_text("P0-SENTINEL\n")
    monkeypatch.setattr(sm, "HERE", ff)
    monkeypatch.setattr(sm, "WORK", tmp_path / "work")
    monkeypatch.setattr(C, "CORPORA", corp)
    monkeypatch.setattr(C, "JOIN_P0_ONLY", False)
    # the script files the record pins: copies, so that the pin test can tamper with them
    for n in ("s28_core.py", "s28_run.py", "run_fitness.py", "flow_decide.py", "flow_p1_audit.py", "flow-brief.template.md"):
        (ff / n).write_bytes((FF / n).read_bytes())
    ext = tmp_path / "extract.sc"
    ext.write_bytes((REAL.parents[1] / "extractors" / "joern" / "extract.sc").read_bytes())
    monkeypatch.setattr(sm, "EXTRACT_SC", ext)
    return ff


def write_record(tmp_path, ff, **over):
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    import run_fitness as _rf
    rec = {"id": sm.RECORD_ID, "corpus": {"commit": COMMIT, "srcDigest": _rf.tree_digest(tmp_path / "corp" / "python" / "src")},
           "coreFreeze": {"s28_core.py": sha(ff / "s28_core.py"), "extract.sc": sha(tmp_path / "extract.sc")},
           "tools": {"s28_run.py": sha(ff / "s28_run.py") + " (cebc4245, run without --guard)", "s28_metrics.py": sha(sm.__file__),
                     "run_fitness.py": sha(ff / "run_fitness.py"), "flow_decide.py": sha(ff / "flow_decide.py"),
                     "flow_p1_audit.py": sha(ff / "flow_p1_audit.py"), "flow-brief.template.md": sha(ff / "flow-brief.template.md")},
           "cpgs": {c: sha(tmp_path / "work" / "python" / f"{c}.cpg.bin") for c in ("off", "on")},
           "shippedFacts": {c: sha(tmp_path / "work" / "python" / "facts.jsonl") for c in ("off", "on")}}
    draw = {"id": sm.RECORD_ID + "-draw", "sampleSha256": sha(ff / "sample" / "python-flow-p1.jsonl")}
    for k, v in over.items():
        a, b = k.split("__")
        if a == "draw":
            draw[b] = v
        else:
            rec[a][b] = v
    p = tmp_path / "record.jsonl"
    p.write_text(json.dumps(rec) + "\n" + json.dumps(draw) + "\n")
    return str(p)


def run_metrics(tmp_path, ff, *args, record=True):
    argv = ["python", "--flow-set", "p1"] + (["--record", write_record(tmp_path, ff)] if record else []) + list(args)
    return sm.main(argv)


def test_p1_run_writes_only_p1_files_and_never_opens_a_p0_file(tmp_path, monkeypatch):
    ff = p1_world(tmp_path, monkeypatch)
    before = tree(tmp_path)
    rec = Recorder(monkeypatch)
    assert run_metrics(tmp_path, ff) == 0
    after = tree(tmp_path)
    new = sorted(set(after) - set(before))
    assert new == ["ff/adjudication/python-flow-p1-misses.jsonl", "ff/adjudication/python-flow-p1-spurious.jsonl",
                   "ff/results/p1/fixed-new/projection-python-p1.answers.jsonl", "ff/results/p1/fixed-new/projection-python-p1.json", "record.jsonl"]
    assert {k for k, v in before.items() if after[k] != v} == set()
    p0 = {"python-flow.jsonl", "python.flow-final.jsonl", "python-flow-audit.jsonl", "projection-python.json", "projection-python.answers.jsonl",
          "python-flow-misses.jsonl", "python-flow-spurious.jsonl"}
    assert not p0 & {Path(p).name for p, _ in rec.opened}                         # not even opened for read
    for w in rec.writes():
        assert "p1" in Path(w).name or Path(w).name == "record.jsonl", w


def test_p1_output_is_labelled_uses_the_p1_spur_seed_and_the_p1_truth(tmp_path, monkeypatch):
    ff = p1_world(tmp_path, monkeypatch)
    run_metrics(tmp_path, ff)
    out = json.loads((ff / "results" / "p1" / "fixed-new" / "projection-python-p1.json").read_text())
    assert out["flowSet"] == "p1" and out["truth"] == "truth/python.flow-p1-final.jsonl"
    assert out["spuriousSample"]["seed"] == f"ccg-p1a-v1-spur|python|{COMMIT}"
    assert out["cell"]["role"] == "decision" and out["cell"]["join"] == "new" and out["cell"]["record"] == sm.RECORD_ID
    assert out["missesFile"]["path"] == "adjudication/python-flow-p1-misses.jsonl"
    assert out["flowAvailableProvisional"]["value"] is not None


def test_p0_run_is_unchanged_in_names_seed_and_keys(tmp_path, monkeypatch):
    ff = p1_world(tmp_path, monkeypatch)
    assert sm.main(["python"]) == 0
    out = json.loads((ff / "results" / "projection-python.json").read_text())
    assert out["truth"] == "truth/python.flow-final.jsonl" and out["spuriousSample"]["seed"] == f"ccg-p0a-v1-spur|python|{COMMIT}"
    assert not {"flowSet", "cell"} & set(out) and out["missesFile"]["path"] == "adjudication/python-flow-misses.jsonl"
    assert (ff / "adjudication" / "python-flow-misses.jsonl").read_text() != "P0-SENTINEL\n"
    assert not (ff / "results" / "p1").exists() and not list((ff / "adjudication").glob("*p1-misses*"))


# sha256 of the default (p0) scoring outputs on the planted world, normalised for the tmp root, as produced by s28_metrics.py at HEAD
# BEFORE P1-34a (cebc4245 + the tree at the start of the unit): P1-34a must not change a p0 byte.
P0_GOLDEN = {"results/projection-python.json": "a8204f8e805779318270c1f539ad86698f753147fa293b9e3674d0da0848d8f1",
             "results/projection-python.answers.jsonl": "709f8db06ce984b62049fa9dafac613491afeb57fd3c7564c62d748e4f31cdfa",
             "adjudication/python-flow-misses.jsonl": "161b6e98d6bb79e90be639381d5cbe807073928185f07421822bbf8b2fb52c7f",
             "adjudication/python-flow-spurious.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}


def test_default_p0_scoring_output_is_byte_identical_to_the_pre_p1_34a_scorer(tmp_path, monkeypatch):
    ff = p1_world(tmp_path, monkeypatch)
    assert sm.main(["python"]) == 0
    got = {n: hashlib.sha256((ff / n).read_text().replace(str(tmp_path), "<T>").encode()).hexdigest() for n in P0_GOLDEN}
    assert got == P0_GOLDEN


def _p0_world_with_spurious_pairs(tmp_path, monkeypatch):
    """The planted world, but the golden set omits four of its connected pairs: the projection still connects them, so they are SPURIOUS, so the spurious
    writer and its seed path run (the full world has none: its spurious golden is the empty file)."""
    ff = p1_world(tmp_path, monkeypatch)
    gold = [json.loads(l) for l in (ff / "truth" / "python.flow-final.jsonl").read_text().splitlines()]
    gold[0]["pairs"] = [pr for i, pr in enumerate(gold[0]["pairs"]) if i not in (0, 2, 3, 5)]       # four connected pairs left out of the golden set
    (ff / "truth" / "python.flow-final.jsonl").write_text("".join(json.dumps(r) + "\n" for r in gold))
    return ff


# sha256 as produced by s28_metrics.py at HEAD before P1-34a (a scratch copy of `git show HEAD:` run through this same test)
P0_SPURIOUS_GOLDEN = {"results/projection-python.json": "a1b046d2d8c4ad270797397b764eeab631f198a1567be60a9b17270ebdbcc1ed",
                      "results/projection-python.answers.jsonl": "90a06255918a3fb788c2faa6178827d70e5799cf9ba842274dbc4c6eb1f71371",
                      "adjudication/python-flow-misses.jsonl": "436bbeebbdbbaea1f03b37f0c9ac0a91a7bae52e7836eaa7f73d10099ad9e0c7",
                      "adjudication/python-flow-spurious.jsonl": "764e36bbcd7e2cdcecb658672b18d71dc5b3e2de2075829f9cbb08a6ce0d1ff5"}


def test_default_p0_scoring_with_spurious_pairs_is_byte_identical_to_the_pre_p1_34a_scorer(tmp_path, monkeypatch):
    ff = _p0_world_with_spurious_pairs(tmp_path, monkeypatch)
    assert sm.main(["python"]) == 0
    sp = (ff / "adjudication" / "python-flow-spurious.jsonl").read_text()
    assert len(sp.splitlines()) >= 3 and sp.strip()                                     # a NON-empty spurious file: the writer ran
    got = {n: hashlib.sha256((ff / n).read_text().replace(str(tmp_path), "<T>").encode()).hexdigest() for n in P0_SPURIOUS_GOLDEN}
    assert got == P0_SPURIOUS_GOLDEN


# The stored real P0 facts, scored by the PRE-P1-34a scorer (`git show HEAD:` copy) at the default join in a scratch HERE holding copies of the P0
# sample/truth/defs/audit files and corpora.yaml: the P1-34a scorer must reproduce these four files byte for byte (answers, misses, spurious equal
# the analyst's review `docs/reviews/connected-code-graph-p1-34a.md` m3; the projection file's hash is this harness's own old-scorer value).
# Opt-in (scores the 37 MB stored facts; ~15 s).
REAL_P0_GOLDEN = {"results/projection-python.json": "4d48b660e034c63fac6b304615e284dd0b281c65f4abb80629b370dfc620247b",
                  "results/projection-python.answers.jsonl": "bd6536ebe1dc08f19fbfdc7cdd188302c6bd42106734cf5ca07d70f4d509b041",
                  "adjudication/python-flow-misses.jsonl": "89018f243b938718a4e9cac2f45431b80631635879a474cb6b06a435d8a8c373",
                  "adjudication/python-flow-spurious.jsonl": "0971e56f5373cb7818e6b5c796d95d26764cfa52ec77b97b3ac64aae8271cf01"}


@pytest.mark.skipif(not os.environ.get("CCG_S28_RECOMPUTE"), reason="UNRUN: opt-in (CCG_S28_RECOMPUTE=1; scores the stored real P0 facts, ~1 min)")
def test_real_p0_facts_at_the_default_join_reproduce_the_pre_p1_34a_scorer_byte_for_byte(tmp_path, monkeypatch):
    work = sm.WORK / "python"
    if not (work / "s28-run.json").exists():
        pytest.skip("the stored P0 run records are not on this machine")
    here = tmp_path / "ff"
    for rel in ("sample/python-flow.jsonl", "truth/python.flow-final.jsonl", "truth/python.defs.jsonl", "adjudication/python-flow-audit.jsonl", "corpora.yaml"):
        (here / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REAL / rel, here / rel)
    (here / "results").mkdir()
    monkeypatch.setattr(sm, "HERE", here)
    monkeypatch.setattr(C, "JOIN_P0_ONLY", False)
    assert sm.main(["python"]) == 0
    assert {n: hashlib.sha256((here / n).read_bytes()).hexdigest() for n in REAL_P0_GOLDEN} == REAL_P0_GOLDEN


def test_the_four_cells_write_distinct_files_and_only_the_decision_cell_writes_adjudication_files(tmp_path, monkeypatch):
    ff = p1_world(tmp_path, monkeypatch)
    run = tmp_path / "work" / "python" / "other-run.json"
    cells = {"decision": [], "fixed/old": ["--join", "p0"], "shipped/new": ["--run-record", str(run)], "shipped/old": ["--run-record", str(run), "--join", "p0"]}
    seen = {}
    for name in ("shipped/old", "shipped/new", "fixed/old", "decision"):           # the decision cell LAST ...
        run_metrics(tmp_path, ff, *cells[name])
        seen[name] = {k: v for k, v in tree(ff).items() if k.startswith(("results/", "adjudication/")) and "p1" in k}
    res = [k for k in seen["decision"] if k.endswith(".json")]
    assert len(res) == 4 and len({k for k in seen["decision"] if k.endswith(".answers.jsonl")}) == 4
    adj = {k for k in seen["decision"] if "misses" in k or "spurious" in k}
    assert adj == {"adjudication/python-flow-p1-misses.jsonl", "adjudication/python-flow-p1-spurious.jsonl"}
    for name in ("shipped/old", "shipped/new", "fixed/old"):                       # ... and no earlier cell wrote an adjudication file
        assert not {k for k in seen[name] if "misses" in k or "spurious" in k}
    # a diagnostic run after the decision cell leaves every decision-cell file and adjudication file byte-identical
    before = {k: v for k, v in tree(ff).items() if k.startswith("results/p1/fixed-new/") or "p1-misses" in k or "p1-spurious" in k}
    assert len(before) == 4
    run_metrics(tmp_path, ff, "--join", "p0")
    after = tree(ff)
    assert all(after[k] == v for k, v in before.items())


def test_no_p1_output_path_of_any_cell_equals_a_p0_path():
    p0 = {str(v) for k, v in sm.flow_paths("python", "p0").items() if k != "spur_prefix"}
    for cell in ("fixed-new", "fixed-p0", "shipped-new", "shipped-p0"):
        p1 = {str(v) for k, v in sm.flow_paths("python", "p1", cell).items() if k != "spur_prefix" and v is not None}
        assert not p0 & p1


@pytest.mark.parametrize("opt", [["--join", "p0"], ["--join", "new"], ["--record", "x"], ["--run-record", "x"]])
def test_p1_options_are_refused_in_p0_mode(tmp_path, monkeypatch, opt):
    ff = p1_world(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="p1 options"):
        sm.main(["python"] + opt)
    assert (ff / "results" / "projection-python.json").read_text() == "P0-SENTINEL\n"


def test_p1_without_a_record_is_refused_and_writes_nothing(tmp_path, monkeypatch):
    ff = p1_world(tmp_path, monkeypatch)
    before = tree(tmp_path)
    with pytest.raises(SystemExit, match="--record"):
        run_metrics(tmp_path, ff, record=False)
    assert tree(tmp_path) == before


def test_join_new_is_the_p1_default_and_p0_join_is_a_labelled_diagnostic(tmp_path, monkeypatch):
    ff = p1_world(tmp_path, monkeypatch)
    seen = []
    real = sm.analyse_cfg
    monkeypatch.setattr(sm, "analyse_cfg", lambda *a, **k: (seen.append(C.JOIN_P0_ONLY), real(*a, **k))[1])
    monkeypatch.setattr(C, "JOIN_P0_ONLY", True)                                   # a stale True must not leak into the decision cell
    run_metrics(tmp_path, ff)
    assert set(seen) == {False}
    seen.clear()
    before = tree(ff)
    run_metrics(tmp_path, ff, "--join", "p0")
    assert set(seen) == {True} and C.JOIN_P0_ONLY is True                         # restored to the pre-run value (the stale True)
    out = json.loads((ff / "results" / "p1" / "fixed-p0" / "projection-python-p1.json").read_text())
    assert out["cell"]["role"].startswith("DIAGNOSTIC") and "old join" in out["variant"] and out["cell"]["join"].startswith("old")
    assert out["flowAvailableProvisional"]["value"] is None
    new = set(tree(ff)) - set(before)
    assert new == {"results/p1/fixed-p0/projection-python-p1.json", "results/p1/fixed-p0/projection-python-p1.answers.jsonl"}
    assert out["cell"]["name"] == "fixed-p0"


def test_a_run_record_without_the_pinned_extract_sha_is_the_shipped_projection_and_a_diagnostic(tmp_path, monkeypatch):
    ff = p1_world(tmp_path, monkeypatch)
    run_metrics(tmp_path, ff, "--run-record", str(tmp_path / "work" / "python" / "other-run.json"))
    out = json.loads((ff / "results" / "p1" / "shipped-new" / "projection-python-p1.json").read_text())
    assert out["cell"]["role"].startswith("DIAGNOSTIC") and out["cell"]["projection"] == "shipped" and out["flowAvailableProvisional"]["value"] is None
    assert not (ff / "results" / "p1" / "fixed-new").exists()
    assert not (ff / "adjudication" / "python-flow-p1-misses.jsonl").exists()


@pytest.mark.parametrize("pin", ["coreFreeze__s28_core.py", "coreFreeze__extract.sc", "tools__s28_run.py", "tools__s28_metrics.py",
                                 "tools__run_fitness.py", "tools__flow_decide.py", "tools__flow_p1_audit.py", "tools__flow-brief.template.md", "draw__sampleSha256",
                                 "corpus__commit", "corpus__srcDigest", "cpgs__off", "cpgs__on"])
def test_a_record_pin_mismatch_refuses_before_anything_is_written(tmp_path, monkeypatch, pin):
    ff = p1_world(tmp_path, monkeypatch)
    before = tree(tmp_path)
    bad = "0" * 64 if pin != "corpus__commit" else "deadbeef"
    assert before
    rec = write_record(tmp_path, ff, **{pin: bad})
    before = tree(tmp_path)
    with pytest.raises(SystemExit, match="NOT-MEASURED:void"):
        sm.main(["python", "--flow-set", "p1", "--record", rec])
    assert tree(tmp_path) == before


SHIPPED = lambda tmp_path: ["--run-record", str(tmp_path / "work" / "python" / "other-run.json")]  # noqa: E731


@pytest.mark.parametrize("extra", [[], "shipped"])
@pytest.mark.parametrize("cfg", ["off", "on"])
def test_a_cpg_rebuilt_after_the_record_is_refused_in_every_cell_and_names_only_the_label(tmp_path, monkeypatch, cfg, extra):
    ff = p1_world(tmp_path, monkeypatch)
    rec = write_record(tmp_path, ff)
    (tmp_path / "work" / "python" / f"{cfg}.cpg.bin").write_bytes(b"A DIFFERENT CPG")
    before = tree(tmp_path)
    args = SHIPPED(tmp_path) if extra else []
    with pytest.raises(SystemExit, match="NOT-MEASURED:void") as ei:
        sm.main(["python", "--flow-set", "p1", "--record", rec] + args)
    assert f"cpgs.{cfg}" in str(ei.value) and tree(tmp_path) == before


@pytest.mark.parametrize("cfg", ["off", "on"])
def test_a_run_record_without_the_cpg_file_or_pointing_at_a_missing_one_is_refused(tmp_path, monkeypatch, cfg):
    ff = p1_world(tmp_path, monkeypatch)
    rec = write_record(tmp_path, ff)
    rr = tmp_path / "work" / "python" / "s28-run.json"
    d = json.loads(rr.read_text())
    for mode in ("absent", "missing-file"):
        e = dict(d["runs"][f"{cfg}.extract"])
        e.pop("cpgFile")
        if mode == "missing-file":
            e["cpgFile"] = str(tmp_path / "nowhere.cpg.bin")
        d2 = json.loads(json.dumps(d))
        d2["runs"][f"{cfg}.extract"] = e
        rr.write_text(json.dumps(d2))
        with pytest.raises(SystemExit, match=f"cpgs.{cfg}"):
            sm.main(["python", "--flow-set", "p1", "--record", rec])


@pytest.mark.parametrize("cfg", ["off", "on"])
def test_shipped_facts_are_pinned_for_the_shipped_projection_only(tmp_path, monkeypatch, cfg):
    ff = p1_world(tmp_path, monkeypatch)
    bad = write_record(tmp_path, ff, **{f"shippedFacts__{cfg}": "0" * 64})
    with pytest.raises(SystemExit, match=f"shippedFacts.{cfg}"):
        sm.main(["python", "--flow-set", "p1", "--record", bad] + SHIPPED(tmp_path))            # the shipped projection: refused
    assert sm.main(["python", "--flow-set", "p1", "--record", bad]) == 0                        # the fixed projection has other facts: not this pin
    good = write_record(tmp_path, ff)
    (tmp_path / "work" / "python" / "facts.jsonl").write_text((tmp_path / "work" / "python" / "facts.jsonl").read_text() + "\n")
    with pytest.raises(SystemExit, match="shippedFacts"):
        sm.main(["python", "--flow-set", "p1", "--record", good] + SHIPPED(tmp_path))           # facts edited after the record


def test_check_record_returns_the_labels_it_checked_including_the_cpg_and_audit_pins(tmp_path, monkeypatch):
    ff = p1_world(tmp_path, monkeypatch)
    rec = write_record(tmp_path, ff)
    runs = json.loads((tmp_path / "work" / "python" / "s28-run.json").read_text())["runs"]
    got = sm.check_record(sm.load_record(rec), sm.load_record(rec, sm.RECORD_ID + "-draw"), "python", COMMIT, ff / "sample" / "python-flow-p1.jsonl", runs)
    assert {"flow_p1_audit.py", "cpgs.off", "cpgs.on"} <= set(got) and not {"shippedFacts.off"} & set(got)


def test_a_record_path_resolves_cwd_relative_and_repo_relative_and_a_missing_one_is_value_free(tmp_path, monkeypatch):
    ff = p1_world(tmp_path, monkeypatch)
    rec = Path(write_record(tmp_path, ff))
    repo = tmp_path / "repo"
    (repo / "cpg" / "eval").mkdir(parents=True)
    (repo / "cpg" / "eval" / "prereg.jsonl").write_text(rec.read_text())
    monkeypatch.setattr(sm, "REPO", repo)
    monkeypatch.chdir(repo / "cpg")                                  # the stated cwd of U7
    for spelling in ("cpg/eval/prereg.jsonl", "eval/prereg.jsonl", str(repo / "cpg" / "eval" / "prereg.jsonl")):
        assert sm.main(["python", "--flow-set", "p1", "--record", spelling]) == 0, spelling
    monkeypatch.chdir(repo)                                          # and from the repo root
    assert sm.main(["python", "--flow-set", "p1", "--record", "cpg/eval/prereg.jsonl"]) == 0
    with pytest.raises(SystemExit, match="not found") as ei:
        sm.main(["python", "--flow-set", "p1", "--record", "SECRETDIR/nope.jsonl"])
    assert "SECRETDIR" not in str(ei.value)


def test_run_record_resolves_repo_relative_too(tmp_path, monkeypatch):
    ff = p1_world(tmp_path, monkeypatch)
    rec = write_record(tmp_path, ff)
    repo = tmp_path / "repo"
    (repo / "cpg").mkdir(parents=True)
    (repo / "cpg" / "run.json").write_text((tmp_path / "work" / "python" / "other-run.json").read_text())
    monkeypatch.setattr(sm, "REPO", repo)
    monkeypatch.chdir(tmp_path)
    assert sm.main(["python", "--flow-set", "p1", "--record", rec, "--run-record", "cpg/run.json"]) == 0
    assert (ff / "results" / "p1" / "shipped-new").exists()


def test_a_placeholder_pin_is_a_mismatch(tmp_path, monkeypatch):
    ff = p1_world(tmp_path, monkeypatch)
    rec = write_record(tmp_path, ff, **{"tools__s28_metrics.py": "<sha at P1-34a commit>"})
    with pytest.raises(SystemExit, match="s28_metrics.py"):
        sm.main(["python", "--flow-set", "p1", "--record", rec])


def test_editing_a_pinned_script_after_the_record_is_refused(tmp_path, monkeypatch):
    ff = p1_world(tmp_path, monkeypatch)
    rec = write_record(tmp_path, ff)
    (ff / "s28_core.py").write_text((ff / "s28_core.py").read_text() + "\n# edited\n")
    with pytest.raises(SystemExit, match="s28_core.py"):
        sm.main(["python", "--flow-set", "p1", "--record", rec])


def test_a_record_may_be_jsonl_lines_among_others_and_must_carry_both_ids(tmp_path, monkeypatch):
    ff = p1_world(tmp_path, monkeypatch)
    rec = Path(write_record(tmp_path, ff))
    jl = tmp_path / "prereg.jsonl"
    jl.write_text(json.dumps({"id": "other"}) + "\n" + rec.read_text())
    assert sm.main(["python", "--flow-set", "p1", "--record", str(jl)]) == 0
    only_v1 = tmp_path / "only.jsonl"
    only_v1.write_text(rec.read_text().splitlines()[0] + "\n")
    with pytest.raises(SystemExit, match="-draw"):
        sm.main(["python", "--flow-set", "p1", "--record", str(only_v1)])
    with pytest.raises(SystemExit, match="no `ccg-p1-flow-python-v1` record"):
        sm.main(["python", "--flow-set", "p1", "--record", str(tmp_path / "work" / "python" / "s28-run.json")])


def test_p1_is_python_only(tmp_path, monkeypatch):
    p1_world(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="python only"):
        sm.main(["go", "--flow-set", "p1", "--record", "x"])


# ------------------------------------------------------------------------------------------------ mutants
MUTANTS = {
    "exclusion_off": ("run_fitness.py", [("    kept = [m for m in base if not _is_excluded(m, excl)]", "    kept = list(base)")]),
    "exclusion_by_file_only": ("run_fitness.py", [("m[\"path\"] == p and m[\"lineStart\"] <= line <= m[\"lineEnd\"]", "m[\"path\"] == p")]),
    "exclusion_by_start_line_only": ("run_fitness.py", [("m[\"path\"] == p and m[\"lineStart\"] <= line <= m[\"lineEnd\"]", "m[\"path\"] == p and m[\"lineStart\"] == line")]),
    "seed_string": ("run_fitness.py", [('"flow": "ccg-p1a-v1-flow|"', '"flow": "ccg-p1a-v2-flow|"')]),
    "spur_seed_string": ("s28_metrics.py", [('"spur_prefix": "ccg-p1a-v1-spur|"', '"spur_prefix": "ccg-p0a-v1-spur|"')]),
    "hash_order_descending": ("run_fitness.py", [("    rows = [{\"ordinal\": i, \"path\": m[\"path\"], \"qname\": m[\"qname\"], \"lineStart\": m[\"lineStart\"], \"lineEnd\": m[\"lineEnd\"],\n             \"source\": m[\"source\"]} for i, (_, m) in enumerate(hash_sort(kept, seed)[:n], start=1)]",
                                                  "    rows = [{\"ordinal\": i, \"path\": m[\"path\"], \"qname\": m[\"qname\"], \"lineStart\": m[\"lineStart\"], \"lineEnd\": m[\"lineEnd\"],\n             \"source\": m[\"source\"]} for i, (_, m) in enumerate(hash_sort(kept, seed)[::-1][:n], start=1)]")]),
    "counts_leak_extra_field": ("run_fitness.py", [('res = {"excludedTier1Frame": k1f, "excludedTier2Frame": k2f}', 'res = {"excludedTier1Frame": k1f, "excludedTier2Frame": k2f, "population": len(pop1)}')]),
    "p0_path_used_in_p1_mode": ("run_fitness.py", [('(_p1_target(flow_path(lang, "p1")), ', '(flow_path(lang, "p0"), ')]),
    "p0_methods_not_removed": ("run_fitness.py", [('    base = [m for m in pop if m["siteKey"] not in drop_keys]', "    base = list(pop)")]),
    "unfiltered_prefix_count_returned": ("run_fitness.py", [('res = {"excludedTier1Frame": k1f, "excludedTier2Frame": k2f}', 'res = {"excludedTier1Frame": k1f, "excludedTier2Frame": k2f, "excludedTier1Prefix30": sum(1 for _, m in hash_sort([m for m in pop1 if m["siteKey"] not in drop], seed)[:30] if _is_excluded(m, excl))}')]),
    "frame_counts_not_asserted": ("run_fitness.py", [("    if got != tuple(expect):", "    if False:")]),
    "default_counts_on_any_corpus": ("run_fitness.py", [("        if not registered:\n            raise SystemExit(\"flowdraw-p1: the registered frame counts", "        if False:\n            raise SystemExit(\"flowdraw-p1: the registered frame counts")]),
    "exclusion_entries_not_validated_against_the_corpus": ("run_fitness.py", [("excl = load_exclusions(exclude, src_dir(lang))", "excl = load_exclusions(exclude)")]),
    "exclusion_path_form_not_strict": ("run_fitness.py", [("    if not _strict_rel(path):\n        return False\n", "")]),
    "exclusion_path_may_leave_the_corpus": ("run_fitness.py", [("if not f.is_file() or src.resolve() not in f.resolve().parents:", "if not f.is_file():")]),
    "exclusion_defline_lower_bound_dropped": ("run_fitness.py", [("    return 1 <= line <= n", "    return line <= n")]),
    "exclusion_defline_zero_allowed": ("run_fitness.py", [("    return 1 <= line <= n", "    return 0 <= line <= n")]),
    "exclusion_defline_upper_bound_dropped": ("run_fitness.py", [("    return 1 <= line <= n", "    return 1 <= line")]),
    "exclusion_defline_last_line_refused": ("run_fitness.py", [("    return 1 <= line <= n", "    return 1 <= line < n")]),
    "exclusion_defline_first_line_refused": ("run_fitness.py", [("    return 1 <= line <= n", "    return 1 < line <= n")]),
    "exclusion_line_count_ignores_a_missing_final_newline": ("run_fitness.py", [('+ (1 if data and not data.endswith(b"\\n") else 0)', "+ 0")]),
    "exclusion_matching_nothing_aborts": ("run_fitness.py", [("    res = {\"excludedTier1Frame\": k1f, \"excludedTier2Frame\": k2f}      # exactly", "    if k1f + k2f == 0:\n        raise SystemExit(\"flowdraw-p1: nothing matched; nothing written\")\n    res = {\"excludedTier1Frame\": k1f, \"excludedTier2Frame\": k2f}      # exactly")]),
    "non_utf8_error_escapes": ("run_fitness.py", [("    except UnicodeDecodeError:", "    except ZeroDivisionError:")]),
    "recursion_error_escapes": ("run_fitness.py", [("except (ValueError, KeyError, RecursionError):", "except (ValueError, KeyError):")]),
    "outputs_written_one_by_one_tier1_first": ("run_fitness.py", [
        ('_write_all([(tier2_out, "".join(jdump(r) + "\\n" for r in rows2)),', '_write_all([(_p1_target(flow_path(lang, "p1")), "".join(jdump(r) + "\\n" for r in rows1)), (tier2_out, "".join(jdump(r) + "\\n" for r in rows2)),'),
        ('                (_p1_target(flow_path(lang, "p1")), "".join(jdump(r) + "\\n" for r in rows1)),\n', ""),
        ("                f.write(text)\n    except OSError:", "                f.write(text)\n            os.replace(tmp, path)\n    except OSError:"),
        ("    for tmp, path in staged:\n        os.replace(tmp, path)", "    pass")]),
    "staging_failure_leaves_temp_files": ("run_fitness.py", [("            tmp.unlink(missing_ok=True)\n", "            pass\n")]),
    "p0_spur_seed_string": ("s28_metrics.py", [('"spur_prefix": "ccg-p0a-v1-spur|"}', '"spur_prefix": "ccg-p0a-v2-spur|"}')]),
    "record_does_not_pin_the_audit_script": ("s28_metrics.py", [('            ("flow_p1_audit.py", get("tools", "flow_p1_audit.py"), HERE / "flow_p1_audit.py"),\n', "")]),
    "cpg_digests_not_checked": ("s28_metrics.py", [('        if not _pin_ok(get("cpgs", c), run.get("cpgFile")):', "        if False:")]),
    "only_the_off_cpg_checked": ("s28_metrics.py", [('        if not _pin_ok(get("cpgs", c), run.get("cpgFile")):', '        if c == "off" and not _pin_ok(get("cpgs", c), run.get("cpgFile")):')]),
    "shipped_facts_not_checked": ("s28_metrics.py", [("        if shipped and not _pin_ok(", "        if False and not _pin_ok(")]),
    "shipped_facts_checked_for_the_fixed_projection_too": ("s28_metrics.py", [("        if shipped and not _pin_ok(", "        if not _pin_ok(")]),
    "pin_without_a_file_passes": ("s28_metrics.py", [("and f is not None and Path(f).is_file() and tok == _file_sha(f)", "and (f is None or tok == _file_sha(f))")]),
    "record_path_cwd_only": ("s28_metrics.py", [("    if not q.is_absolute() and (REPO / q).exists():", "    if False:")]),
    "record_path_repo_only": ("s28_metrics.py", [("    if q.exists():\n        return q\n", "")]),
    "record_path_error_echoes_the_path": ("s28_metrics.py", [('raise SystemExit(f"{q.name}: not found', 'raise SystemExit(f"{q}: not found')]),
    "run_record_not_resolved": ("s28_metrics.py", [("run_file = resolve_path(run_path) if a.run_record else run_path", "run_file = run_path")]),
    "empty_exclusion_allowed": ("run_fitness.py", [("    if not excl:\n", "    if False:\n")]),
    "tier2_written_into_the_repo_sample_dir": ("run_fitness.py", [("_write_all([(tier2_out, ", '_write_all([(flow_tier2_path(lang, "p1"), ')]),
    "tier2_out_inside_repo_allowed": ("run_fitness.py", [("    if REPO in (tier2_out.resolve(), *tier2_out.resolve().parents):", "    if False:")]),
    "registered_counts_wrong": ("run_fitness.py", [("P1_FRAME_COUNTS = (368, 338, 163)", "P1_FRAME_COUNTS = (368, 338, 164)")]),
    "exclusion_path_echoed": ("run_fitness.py", [('raise SystemExit("exclusion file: cannot be read") from None', 'raise SystemExit(f"exclusion file: cannot read {path}")')]),
    "exclusion_line_echoed": ("run_fitness.py", [('raise SystemExit(f"exclusion file: line {no} is not', 'raise SystemExit(f"exclusion file: line {no} {ln} is not')]),
    "metrics_p1_results_path_is_p0": ("s28_metrics.py", [('"results": rdir / f"projection-{lang}-p1.json"', '"results": HERE / "results" / f"projection-{lang}.json"')]),
    "metrics_cells_share_a_directory": ("s28_metrics.py", [('rdir = HERE / "results" / "p1" / cell', 'rdir = HERE / "results" / "p1"')]),
    "metrics_every_cell_writes_adjudication": ("s28_metrics.py", [("if cell == DECISION_CELL else (lambda n: None)", "if True else (lambda n: None)")]),
    "metrics_projection_label_always_fixed": ("s28_metrics.py", [('    return "fixed" if all(', '    return "fixed" if True or all(')]),
    "record_does_not_pin_decide_script": ("s28_metrics.py", [('            ("flow_decide.py", get("tools", "flow_decide.py"), HERE / "flow_decide.py"),\n', "")]),
    "record_does_not_pin_corpus_digest": ("s28_metrics.py", [('    if get("corpus", "srcDigest") != rf.tree_digest(C.CORPORA / lang / "src"):', "    if False:")]),
    "draw_record_not_required": ("s28_metrics.py", [('check_record(rec, load_record(a.record, RECORD_ID + "-draw"), lang,', 'check_record(rec, {"sampleSha256": _file_sha(P["sample"])}, lang,')]),
    "seed_guard_removed": ("run_fitness.py", [("P1_REAL_COMMIT, \"p1\") and not registered:", "P1_REAL_COMMIT, \"p1\") and False:")]),
    "counts_gain_digests": ("run_fitness.py", [('res = {"excludedTier1Frame": k1f, "excludedTier2Frame": k2f}', 'res = {"excludedTier1Frame": k1f, "excludedTier2Frame": k2f, "sampleSha256": hashlib.sha256(json.dumps(rows1).encode()).hexdigest()}')]),
    "tier2_uses_tier1_bounds": ("run_fitness.py", [("pop2 = flow_population(lang, src_dir(lang), cfg, 26, 60)", "pop2 = flow_population(lang, src_dir(lang), cfg, 5, 25)")]),
    "metrics_p1_truth_is_p0": ("s28_metrics.py", [('"truth": HERE / "truth" / f"{lang}.flow-p1-final.jsonl"', '"truth": HERE / "truth" / f"{lang}.flow-final.jsonl"')]),
    "metrics_p1_misses_is_p0": ("s28_metrics.py", [('adj = (lambda n: HERE / "adjudication" / f"{lang}-flow-p1-{n}.jsonl")', 'adj = (lambda n: HERE / "adjudication" / f"{lang}-flow-{n}.jsonl")')]),
    "metrics_p1_audit_is_p0": ("s28_metrics.py", [('"audit": HERE / "adjudication" / f"{lang}-flow-p1-audit.jsonl"', '"audit": HERE / "adjudication" / f"{lang}-flow-audit.jsonl"')]),
    "record_check_removed": ("s28_metrics.py", [("        check_record(rec, load_record(a.record, RECORD_ID + \"-draw\"), lang,", "        (lambda *x: None)(rec, load_record(a.record, RECORD_ID + \"-draw\"), lang,")]),
    "record_does_not_pin_metrics_itself": ("s28_metrics.py", [('            ("s28_metrics.py", get("tools", "s28_metrics.py"), Path(__file__).resolve()),\n', "")]),
    "record_does_not_pin_core": ("s28_metrics.py", [('("s28_core.py", get("coreFreeze", "s28_core.py"), HERE / "s28_core.py"),\n', "")]),
    "record_sha_compare_removed": ("s28_metrics.py", [("and f is not None and Path(f).is_file() and tok == _file_sha(f)", "and f is not None and Path(f).is_file()")]),
    "join_default_old": ("s28_metrics.py", [('join = a.join or "new"', 'join = a.join or "p0"')]),
    "join_switch_ignored": ("s28_metrics.py", [('C.JOIN_P0_ONLY = join == "p0"', "C.JOIN_P0_ONLY = False")]),
    "diag_unlabelled": ("s28_metrics.py", [('"value": None if diag else bool(asshipped_ok and not guard)', '"value": bool(asshipped_ok and not guard)')]),
    "join_not_restored": ("s28_metrics.py", [("        C.JOIN_P0_ONLY = saved_join", "        pass")]),
}


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_is_killed(name, tmp_path):
    if os.environ.get("CCG_FF_DIR"):
        pytest.skip("inner run")
    copy = tmp_path / "src"
    copy.mkdir()
    for p in REAL.glob("*.py"):
        shutil.copy(p, copy / p.name)
    for n in ("corpora.yaml", "flow-brief.template.md"):
        shutil.copy(REAL / n, copy / n)
    fname, edits = MUTANTS[name]
    text = (copy / fname).read_text(encoding="utf-8")
    for old, new in edits:
        assert old in text, f"mutant {name}: pattern not found, the mutant would be a no-op"
        text = text.replace(old, new)
    (copy / fname).write_text(text, encoding="utf-8")
    env = dict(os.environ, CCG_FF_DIR=str(copy), CENSUS_SRC=str(copy), PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-k", "not mutant", str(Path(__file__))],
                       capture_output=True, text=True, env=env, cwd=Path(__file__).resolve().parents[1])
    assert p.returncode != 0 and "failed" in p.stdout, f"mutant {name} survived:\n{p.stdout[-600:]}"
